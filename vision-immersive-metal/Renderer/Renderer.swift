//
//  Renderer.swift
//

import CompositorServices
import Metal
import MetalKit
import simd
import AVFoundation

// The 256 byte aligned size of our uniform structure
nonisolated let alignedUniformsSize = (MemoryLayout<Uniforms>.size + 0xFF) & -0x100
nonisolated let alignedViewProjectionArraySize = (MemoryLayout<ViewProjectionArray>.size + 0xFF) & -0x100

nonisolated let maxBuffersInFlight = 1

enum RendererError: Error {
    case badVertexDescriptor
}

extension MTLDevice {
    nonisolated var supportsMSAA: Bool {
        supports32BitMSAA && supportsTextureSampleCount(4)
    }

    nonisolated var rasterSampleCount: Int {
        supportsMSAA ? 4 : 1
    }
}

extension LayerRenderer.Clock.Instant {
    nonisolated var timeInterval: TimeInterval {
        let components = LayerRenderer.Clock.Instant.epoch.duration(to: self).components
        let nanoseconds = TimeInterval(components.attoseconds / 1_000_000_000)
        return TimeInterval(components.seconds) + (nanoseconds / TimeInterval(NSEC_PER_SEC))
    }
}

final class RendererTaskExecutor: TaskExecutor {
    private let queue = DispatchQueue(label: "RenderThreadQueue", qos: .userInteractive)

    func enqueue(_ job: UnownedJob) {
        queue.async {
          job.runSynchronously(on: self.asUnownedSerialExecutor())
        }
    }

    nonisolated func asUnownedSerialExecutor() -> UnownedTaskExecutor {
        return UnownedTaskExecutor(ordinary: self)
    }

    static var shared: RendererTaskExecutor = RendererTaskExecutor()
}


actor Renderer {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    #if !targetEnvironment(simulator)
    let residencySets: [MTLResidencySet]
    let commandQueueResidencySet: MTLResidencySet
    #endif

    let dynamicUniformBuffer: MTLBuffer
    let pipelineState: MTLRenderPipelineState
    let depthState: MTLDepthStencilState
    let textureCache: CVMetalTextureCache
    var colorMap: MTLTexture

    let endFrameEvent: MTLSharedEvent
    var committedFrameIndex: UInt64 = 0

    var uniformBufferOffset = 0

    var uniformBufferIndex = 0

    var uniforms: UnsafeMutablePointer<Uniforms>

    var perDrawableTarget = [LayerRenderer.Drawable.Target: DrawableTarget]()

    var rotation: Float = 0

    var mesh: MTKMesh

    let worldTracking: WorldTrackingProvider
    let layerRenderer: LayerRenderer
    
    let appModel: AppModel
    
    var videoOutput: AVPlayerItemVideoOutput?
    private var textureMappingsInFlight: Set<CVMetalTexture> = []
    private var lastFrameTexture: MTLTexture?

    init(_ layerRenderer: LayerRenderer, appModel: AppModel, videoOutput: AVPlayerItemVideoOutput) {
        self.layerRenderer = layerRenderer
        self.device = layerRenderer.device
        self.appModel = appModel
        self.videoOutput = videoOutput

        let device = self.device
        self.commandQueue = self.device.makeCommandQueue()!
        
        var textureCache: CVMetalTextureCache!
        let _ = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        self.textureCache = textureCache

        #if !targetEnvironment(simulator)
        let residencySetDesc = MTLResidencySetDescriptor()
        residencySetDesc.initialCapacity = 3 // color + depth + view projection buffer
        self.residencySets = (0...maxBuffersInFlight).map { _ in try! device.makeResidencySet(descriptor: residencySetDesc) }
        #endif

        self.endFrameEvent = device.makeSharedEvent()!
        // Start the signal value + committed frames index at
        // max buffers in flight to avoid negative values
        self.endFrameEvent.signaledValue = UInt64(maxBuffersInFlight)
        committedFrameIndex = UInt64(maxBuffersInFlight)

        let uniformBufferSize = alignedUniformsSize * maxBuffersInFlight

        self.dynamicUniformBuffer = self.device.makeBuffer(length: uniformBufferSize,
                                                           options: [MTLResourceOptions.storageModeShared])!

        self.dynamicUniformBuffer.label = "UniformBuffer"

        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(to: Uniforms.self, capacity: 1)

        let mtlVertexDescriptor = Self.buildMetalVertexDescriptor()

        do {
            pipelineState = try Self.buildRenderPipeline(device: device,
                                                         layerRenderer: layerRenderer,
                                                         mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            fatalError("Unable to compile render pipeline state.  Error info: \(error)")
        }

        self.depthState = Self.buildDepthStencilState(device: device)

        do {
            mesh = try Self.buildMesh(device: device, mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            fatalError("Unable to build MetalKit Mesh. Error info: \(error)")
        }

        do {
            colorMap = try Self.loadTexture(device: device, textureName: "SBS")
        } catch {
            fatalError("Unable to load texture. Error info: \(error)")
        }

        #if !targetEnvironment(simulator)
        // Add all persistent resources to the command queue residency set,
        // must be done after loading all resources.
        residencySetDesc.initialCapacity = mesh.vertexBuffers.count + mesh.submeshes.count + 2 // color map + uniforms buffer
        let residencySet = try! self.device.makeResidencySet(descriptor: residencySetDesc)
        residencySet.addAllocations(mesh.vertexBuffers.map { $0.buffer })
        residencySet.addAllocations(mesh.submeshes.map { $0.indexBuffer.buffer })
        residencySet.addAllocations([colorMap, dynamicUniformBuffer])
        residencySet.commit()
        commandQueueResidencySet = residencySet
        commandQueue.addResidencySet(residencySet)
        #endif

        worldTracking = WorldTrackingProvider()
    }

    private func startARSession(_ arSession: ARKitSession) async {
        do {
            try await arSession.run([worldTracking])
        } catch {
            fatalError("Failed to initialize ARSession")
        }
    }

    @MainActor
    static func startRenderLoop(_ layerRenderer: LayerRenderer, appModel: AppModel, arSession: ARKitSession) {
        Task(executorPreference: RendererTaskExecutor.shared) { @MainActor in
            let renderer = Renderer(layerRenderer, appModel: appModel, videoOutput: appModel.videoOutput!)
            await renderer.startARSession(arSession)
            await renderer.renderLoop()
        }
    }

    static func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
        // Create a Metal vertex descriptor specifying how vertices will by laid out for input into our render
        //   pipeline and how we'll layout our Model IO vertices

        let mtlVertexDescriptor = MTLVertexDescriptor()

        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue

        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].format = MTLVertexFormat.float2
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].bufferIndex = BufferIndex.meshGenerics.rawValue

        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = MemoryLayout<SIMD3<Float>>.stride
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stride = MemoryLayout<SIMD2<Float>>.stride
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        return mtlVertexDescriptor
    }

    static func buildRenderPipeline(device: MTLDevice,
                                    layerRenderer: LayerRenderer,
                                    mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object

        let library = device.makeDefaultLibrary()

        let vertexFunction = library?.makeFunction(name: "vertexShader")
        let fragmentFunction = library?.makeFunction(name: "fragmentShader")

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor
        pipelineDescriptor.rasterSampleCount = device.rasterSampleCount

        pipelineDescriptor.colorAttachments[0].pixelFormat = layerRenderer.configuration.colorFormat
        pipelineDescriptor.depthAttachmentPixelFormat = layerRenderer.configuration.depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = layerRenderer.properties.viewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    static func buildDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.greater
        depthStateDescriptor.isDepthWriteEnabled = true
        return device.makeDepthStencilState(descriptor: depthStateDescriptor)!
    }

    static func buildMesh(device: MTLDevice,
                          mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTKMesh {
        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor

        let metalAllocator = MTKMeshBufferAllocator(device: device)

        // Create custom hemisphere mesh for VR
        let mdlMesh = try Self.buildHemisphereMesh(allocator: metalAllocator)

        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)

        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
            throw RendererError.badVertexDescriptor
        }
        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate

        mdlMesh.vertexDescriptor = mdlVertexDescriptor

        return try MTKMesh(mesh: mdlMesh, device: device)
    }

    static func buildHemisphereMesh(allocator: MTKMeshBufferAllocator) throws -> MDLMesh {
        let radius: Float = 1000.0
        let radialSegments: Int = 128   // Horizontal resolution (azimuth)
        let verticalSegments: Int = 64   // Vertical resolution (elevation)

        // Generate vertices for front hemisphere only (-π/2 to π/2 azimuth)
        var vertices: [SIMD3<Float>] = []
        var texcoords: [SIMD2<Float>] = []

        // Generate hemisphere vertices
        for y in 0...verticalSegments {
            // Elevation: -π/2 (bottom) to π/2 (top)
            let elevation = Float.pi * (Float(y) / Float(verticalSegments) - 0.5)

            for x in 0...radialSegments {
                // Azimuth: -π/2 (left) to π/2 (right) - front hemisphere only
                let azimuth = Float.pi * (Float(x) / Float(radialSegments) - 0.5)

                // Convert spherical to Cartesian coordinates
                let cosElev = cos(elevation)
                let sinElev = sin(elevation)
                let cosAzim = cos(azimuth)
                let sinAzim = sin(azimuth)

                // Position: hemisphere facing forward (+Z direction), but flip Z for interior viewing
                let pos = SIMD3<Float>(
                    radius * cosElev * sinAzim,  // X (left-right)
                    radius * sinElev,            // Y (up-down)
                    -radius * cosElev * cosAzim  // -Z (facing viewer)
                )
                vertices.append(pos)

                // UV coordinates for VR180 content
                // Map azimuth (-π/2 to π/2) to U (0 to 0.5)
                let u = Float(x) / Float(radialSegments) * 0.5
                // Map elevation (-π/2 to π/2) to V (0 to 1)
                let v = Float(y) / Float(verticalSegments)

                texcoords.append(SIMD2<Float>(u, v))
            }
        }

        // Generate triangle indices
        var indices: [UInt32] = []
        for y in 0..<verticalSegments {
            for x in 0..<radialSegments {
                let current = y * (radialSegments + 1) + x
                let next = current + 1
                let bottom = (y + 1) * (radialSegments + 1) + x
                let bottomNext = bottom + 1

                // First triangle
                indices.append(UInt32(current))
                indices.append(UInt32(bottom))
                indices.append(UInt32(next))

                // Second triangle
                indices.append(UInt32(next))
                indices.append(UInt32(bottom))
                indices.append(UInt32(bottomNext))
            }
        }

        // Create mesh data using MTKMeshBufferAllocator
        let vertexBuffer = allocator.newBuffer(from: nil, length: MemoryLayout<SIMD3<Float>>.size * vertices.count, type: .vertex)!
        let texcoordBuffer = allocator.newBuffer(from: nil, length: MemoryLayout<SIMD2<Float>>.size * texcoords.count, type: .vertex)!
        let indexBuffer = allocator.newBuffer(from: nil, length: MemoryLayout<UInt32>.size * indices.count, type: .index)!

        // Copy data to buffers
        let vertexMap = vertexBuffer.map()
        memcpy(vertexMap.bytes, vertices, MemoryLayout<SIMD3<Float>>.size * vertices.count)

        let texcoordMap = texcoordBuffer.map()
        memcpy(texcoordMap.bytes, texcoords, MemoryLayout<SIMD2<Float>>.size * texcoords.count)

        let indexMap = indexBuffer.map()
        memcpy(indexMap.bytes, indices, MemoryLayout<UInt32>.size * indices.count)

        let vertexSource = vertexBuffer
        let texcoordSource = texcoordBuffer
        let indexSource = indexBuffer

        // Create vertex descriptor
        let vertexDescriptor = MDLVertexDescriptor()

        // Position attribute (buffer index 0)
        let positionAttribute = MDLVertexAttribute(name: MDLVertexAttributePosition, format: .float3, offset: 0, bufferIndex: 0)
        let positionLayout = MDLVertexBufferLayout(stride: MemoryLayout<SIMD3<Float>>.stride)
        vertexDescriptor.attributes = [positionAttribute]
        vertexDescriptor.layouts = [positionLayout]

        // Texture coordinate attribute (buffer index 1)
        let texcoordAttribute = MDLVertexAttribute(name: MDLVertexAttributeTextureCoordinate, format: .float2, offset: 0, bufferIndex: 1)
        let texcoordLayout = MDLVertexBufferLayout(stride: MemoryLayout<SIMD2<Float>>.stride)
        vertexDescriptor.attributes.add(texcoordAttribute)
        vertexDescriptor.layouts.add(texcoordLayout)

        // Create submesh
        let submesh = MDLSubmesh(indexBuffer: indexSource, indexCount: indices.count, indexType: .uint32, geometryType: .triangles, material: nil)

        // Create mesh
        let mesh = MDLMesh(vertexBuffers: [vertexSource, texcoordSource], vertexCount: vertices.count, descriptor: vertexDescriptor, submeshes: [submesh])

        return mesh
    }

    static func loadTexture(device: MTLDevice,
                            textureName: String) throws -> MTLTexture {
        /// Load texture data with optimal parameters for sampling

        let textureLoader = MTKTextureLoader(device: device)

        let textureLoaderOptions = [
            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue)
        ]

        return try textureLoader.newTexture(name: textureName,
                                            scaleFactor: 1.0,
                                            bundle: nil,
                                            options: textureLoaderOptions)
    }

    private func updateDynamicBufferState(frameIndex: UInt64) {
        /// Update the state of our uniform buffers before rendering

        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight

        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex

        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to: Uniforms.self, capacity: 1)

        /// Reset resources used in previous frame

        #if !targetEnvironment(simulator)
        residencySets[uniformBufferIndex].removeAllAllocations()
        residencySets[uniformBufferIndex].commit()
        #endif

        /// Remove all per drawable target resources that are older than 90 frames

        perDrawableTarget = perDrawableTarget.filter { $0.value.lastUsedFrameIndex + 90 > frameIndex }
    }

    private func updateGameState() {
        /// Update any game state before rendering

//        let rotationAxis = SIMD3<Float>(1, 0, 0)
//        let modelRotationMatrix = matrix4x4_rotation(radians: 3.141592653589*2, axis: rotationAxis)
        let modelTranslationMatrix = matrix4x4_translation(0.0, 0.0, 0.0)
        let modelMatrix = modelTranslationMatrix //* modelRotationMatrix

        self.uniforms[0].modelMatrix = modelMatrix

        rotation += 0.01
    }
    
    private func extractFrameTexture() -> CVMetalTexture? {
        guard let videoOutput = self.videoOutput else {
            print("videoOutput is nil")
            return nil
        }
        
        let itemTime = videoOutput.itemTime(forHostTime: CACurrentMediaTime())
        if !videoOutput.hasNewPixelBuffer(forItemTime: itemTime) {
            print("No new pixel buffer available for media time: \(itemTime)")
            return nil
        }
        
        guard let pixelBuf = videoOutput.copyPixelBuffer(forItemTime: itemTime,
                                                         itemTimeForDisplay: nil) else {
            print("Failed to copy pixel buffer")
            return nil
        }
        let texWidth = CVPixelBufferGetWidth(pixelBuf)
        let texHeight = CVPixelBufferGetHeight(pixelBuf)
        
        var cvTexture: CVMetalTexture?
        let _ = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                          textureCache,
                                                          pixelBuf,
                                                          nil,
                                                          layerRenderer.configuration.colorFormat,
                                                          texWidth, texHeight, 0, &cvTexture)
        
        return cvTexture
    }

    func renderFrame() {
        /// Per frame updates hare

        guard let frame = layerRenderer.queryNextFrame() else { return }

        guard self.endFrameEvent.wait(untilSignaledValue: committedFrameIndex - UInt64(maxBuffersInFlight), timeoutMS: 10000) else {
            return
        }

        frame.startUpdate()

        // Perform frame independent work

        self.updateDynamicBufferState(frameIndex: frame.frameIndex)

        self.updateGameState()

        let cvTexture = extractFrameTexture()
        var frameTexture: MTLTexture? = lastFrameTexture ?? colorMap
        if cvTexture != nil {
            frameTexture = CVMetalTextureGetTexture(cvTexture!)
        } else {
            print("No texture extracted!")
        }
        if frameTexture == nil {
            print("Failed to convert MTLTexture from pixel buffer!")
            frameTexture = lastFrameTexture ?? colorMap
        } else {
            lastFrameTexture = frameTexture
        }

        frame.endUpdate()
        
        guard let timing = frame.predictTiming() else { return }
        LayerRenderer.Clock().wait(until: timing.optimalInputTime)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Failed to create command buffer")
        }

        #if !targetEnvironment(simulator)
        commandBuffer.useResidencySet(self.residencySets[uniformBufferIndex])
        #endif

        let drawables = frame.queryDrawables()
        guard !drawables.isEmpty else { return }

        frame.startSubmission()

        for drawable in drawables {
            render(frameTexture: frameTexture!, drawable: drawable, commandBuffer: commandBuffer, frameIndex: frame.frameIndex)
        }

        committedFrameIndex += 1

        commandBuffer.encodeSignalEvent(self.endFrameEvent, value: committedFrameIndex)
        
//        if cvTexture != nil {
//            textureMappingsInFlight.insert(cvTexture!)
//            commandBuffer.addCompletedHandler { [weak self] _ in
//                Task.init {
//                    self?.removeTextureMapping(cvTexture!)
//                }
//            }
//        }

        commandBuffer.commit()

        frame.endSubmission()
    }

    func render(frameTexture: MTLTexture, drawable: LayerRenderer.Drawable, commandBuffer: MTLCommandBuffer, frameIndex: UInt64) {
        let time = drawable.frameTiming.presentationTime.timeInterval
        let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: time)

        drawable.deviceAnchor = deviceAnchor

        if perDrawableTarget[drawable.target] == nil {
            perDrawableTarget[drawable.target] = .init(drawable: drawable)
        }
        let drawableTarget = perDrawableTarget[drawable.target]!

        drawableTarget.updateBufferState(uniformBufferIndex: uniformBufferIndex, frameIndex: frameIndex)

        drawableTarget.updateViewProjectionArray(drawable: drawable)

        let renderPassDescriptor = MTLRenderPassDescriptor()

        if device.supportsMSAA {
            let renderTargets = drawableTarget.memorylessTargets[uniformBufferIndex]
            assert(renderTargets.color.width == drawable.colorTextures[0].width)
            assert(renderTargets.color.height == drawable.colorTextures[0].height)

            renderPassDescriptor.colorAttachments[0].resolveTexture = drawable.colorTextures[0]
            renderPassDescriptor.colorAttachments[0].texture = renderTargets.color
            renderPassDescriptor.depthAttachment.resolveTexture = drawable.depthTextures[0]
            renderPassDescriptor.depthAttachment.texture = renderTargets.depth

            renderPassDescriptor.colorAttachments[0].storeAction = .multisampleResolve
            renderPassDescriptor.depthAttachment.storeAction = .multisampleResolve
        } else {
            renderPassDescriptor.colorAttachments[0].texture = drawable.colorTextures[0]
            renderPassDescriptor.depthAttachment.texture = drawable.depthTextures[0]

            renderPassDescriptor.colorAttachments[0].storeAction = .store
            renderPassDescriptor.depthAttachment.storeAction = .store
        }

        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)
        renderPassDescriptor.depthAttachment.loadAction = .clear
        renderPassDescriptor.depthAttachment.clearDepth = 0.0
        renderPassDescriptor.rasterizationRateMap = drawable.rasterizationRateMaps.first
        if layerRenderer.configuration.layout == .layered {
            renderPassDescriptor.renderTargetArrayLength = drawable.views.count
        }

        #if !targetEnvironment(simulator)
        let residencySet = self.residencySets[uniformBufferIndex]
        residencySet.addAllocations([
            drawable.colorTextures[0],
            drawable.depthTextures[0],
            drawableTarget.viewProjectionBuffer
        ])
        residencySet.commit()
        #endif

        /// Final pass rendering code here
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create render encoder")
        }

        renderEncoder.label = "Primary Render Encoder"

        renderEncoder.pushDebugGroup("Draw Box")

//        renderEncoder.setCullMode(.back)
//
//        renderEncoder.setFrontFacing(.counterClockwise)

        renderEncoder.setRenderPipelineState(pipelineState)

        renderEncoder.setDepthStencilState(depthState)

        let viewports = drawable.views.map { $0.textureMap.viewport }

        renderEncoder.setViewports(viewports)

        if drawable.views.count > 1 {
            var viewMappings = (0..<drawable.views.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
        }

        renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)

        renderEncoder.setVertexBuffer(drawableTarget.viewProjectionBuffer, offset: drawableTarget.viewProjectionBufferOffset, index: BufferIndex.viewProjection.rawValue)

        for (index, element) in mesh.vertexDescriptor.layouts.enumerated() {
            guard let layout = element as? MDLVertexBufferLayout else {
                return
            }

            if layout.stride != 0 {
                let buffer = mesh.vertexBuffers[index]
                renderEncoder.setVertexBuffer(buffer.buffer, offset: buffer.offset, index: index)
            }
        }

        renderEncoder.setFragmentTexture(frameTexture, index: TextureIndex.color.rawValue)

        for submesh in mesh.submeshes {
            renderEncoder.drawIndexedPrimitives(type: submesh.primitiveType,
                                                indexCount: submesh.indexCount,
                                                indexType: submesh.indexType,
                                                indexBuffer: submesh.indexBuffer.buffer,
                                                indexBufferOffset: submesh.indexBuffer.offset)
        }

        renderEncoder.popDebugGroup()

        renderEncoder.endEncoding()

        drawable.encodePresent(commandBuffer: commandBuffer)
    }

    func renderLoop() {
        while true {
            if layerRenderer.state == .invalidated {
                print("Layer is invalidated")
                Task { @MainActor in
                    appModel.immersiveSpaceState = .closed
                }
                return
            } else if layerRenderer.state == .paused {
                Task { @MainActor in
                    appModel.immersiveSpaceState = .inTransition
                }
                layerRenderer.waitUntilRunning()
                continue
            } else {
                Task { @MainActor in
                    if appModel.immersiveSpaceState != .open {
                        appModel.immersiveSpaceState = .open
                    }
                }
                autoreleasepool {
                    self.renderFrame()
                }
            }
        }
    }
}

extension Renderer {
    func removeTextureMapping(_ cvTexture: CVMetalTexture) {
        self.textureMappingsInFlight.remove(cvTexture)
    }
}

extension Renderer {
    class DrawableTarget {
        var lastUsedFrameIndex: UInt64

        let memorylessTargets: [(color: MTLTexture, depth: MTLTexture)]

        let viewProjectionBuffer: MTLBuffer

        var viewProjectionBufferOffset = 0

        var viewProjectionArray: UnsafeMutablePointer<ViewProjectionArray>

        nonisolated init(drawable: LayerRenderer.Drawable) {
            lastUsedFrameIndex = 0

            let device = drawable.colorTextures[0].device
            nonisolated func renderTarget(resolveTexture: MTLTexture) -> MTLTexture {
                assert(device.supportsMSAA)

                let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: resolveTexture.pixelFormat,
                                                                          width: resolveTexture.width,
                                                                          height: resolveTexture.height,
                                                                          mipmapped: false)
                descriptor.usage = .renderTarget
                descriptor.textureType = .type2DMultisampleArray
                descriptor.sampleCount = device.rasterSampleCount
                descriptor.storageMode = .memoryless
                descriptor.arrayLength = resolveTexture.arrayLength
                return device.makeTexture(descriptor: descriptor)!
            }

            if device.supportsMSAA {
                memorylessTargets = .init(repeating: (renderTarget(resolveTexture: drawable.colorTextures[0]),
                                                      renderTarget(resolveTexture: drawable.depthTextures[0])),
                                          count: maxBuffersInFlight)
            } else {
                memorylessTargets = []
            }

            let bufferSize = alignedViewProjectionArraySize * maxBuffersInFlight

            viewProjectionBuffer = device.makeBuffer(length: bufferSize,
                                                     options: [MTLResourceOptions.storageModeShared])!
            viewProjectionArray = UnsafeMutableRawPointer(viewProjectionBuffer.contents() + viewProjectionBufferOffset).bindMemory(to: ViewProjectionArray.self, capacity: 1)
        }
    }
}

extension Renderer.DrawableTarget {
    nonisolated func updateBufferState(uniformBufferIndex: Int, frameIndex: UInt64) {
        viewProjectionBufferOffset = alignedViewProjectionArraySize * uniformBufferIndex

        viewProjectionArray = UnsafeMutableRawPointer(viewProjectionBuffer.contents() + viewProjectionBufferOffset).bindMemory(to: ViewProjectionArray.self, capacity: 1)

        lastUsedFrameIndex = frameIndex
    }

    nonisolated func updateViewProjectionArray(drawable: LayerRenderer.Drawable) {
        let simdDeviceAnchor = drawable.deviceAnchor?.originFromAnchorTransform ?? matrix_identity_float4x4

        nonisolated func viewProjection(forViewIndex viewIndex: Int) -> float4x4 {
            let view = drawable.views[viewIndex]
            let viewMatrix = (simdDeviceAnchor * view.transform).inverse
            let projectionMatrix = drawable.computeProjection(viewIndex: viewIndex)

            return projectionMatrix * viewMatrix
        }

        viewProjectionArray[0].viewProjectionMatrix.0 = viewProjection(forViewIndex: 0)
        if drawable.views.count > 1 {
            viewProjectionArray[0].viewProjectionMatrix.1 = viewProjection(forViewIndex: 1)
        }
    }
}

// Generic matrix math utility functions
nonisolated func matrix4x4_rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    return .init(columns: (vector_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                           vector_float4(x * y * ci - z * st, ct + y * y * ci, z * y * ci + x * st, 0),
                           vector_float4(x * z * ci + y * st, y * z * ci - x * st, ct + z * z * ci, 0),
                           vector_float4(                  0, 0, 0, 1)))
}

nonisolated func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float) -> matrix_float4x4 {
    return .init(columns: (vector_float4(1, 0, 0, 0),
                           vector_float4(0, 1, 0, 0),
                           vector_float4(0, 0, 1, 0),
                           vector_float4(translationX, translationY, translationZ, 1)))
}
