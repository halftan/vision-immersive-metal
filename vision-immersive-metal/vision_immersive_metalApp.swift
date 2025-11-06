//
//  vision_immersive_metalApp.swift
//  vision-immersive-metal
//
//

import ARKit
import CompositorServices
import SwiftUI
import AVFoundation

struct ImmersiveSpaceContent: CompositorContent {

    var appModel: AppModel

    var body: some CompositorContent {
        CompositorLayer(configuration: self) { @MainActor layerRenderer in
//            let url = Bundle.main.url(forResource: "VR180test", withExtension: "mov")!
            Renderer.startRenderLoop(layerRenderer, appModel: appModel, arSession: ARKitSession())
        }
    }
}

extension ImmersiveSpaceContent: CompositorLayerConfiguration {
    func makeConfiguration(capabilities: LayerRenderer.Capabilities, configuration: inout LayerRenderer.Configuration) {
        let foveationEnabled = capabilities.supportsFoveation
        configuration.isFoveationEnabled = foveationEnabled

        let options: LayerRenderer.Capabilities.SupportedLayoutsOptions = foveationEnabled ? [.foveationEnabled] : []
        let supportedLayouts = capabilities.supportedLayouts(options: options)

        configuration.layout = supportedLayouts.contains(.layered) ? .layered : .dedicated
        configuration.depthFormat = .depth32Float
        configuration.colorFormat = .rgba16Float
    }
}

@main
struct vision_immersive_metalApp: App {

    @State private var appModel = AppModel()


    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(appModel)
        }

        ImmersiveSpace(id: appModel.immersiveSpaceID) {
            ImmersiveSpaceContent(appModel: appModel)
                .onAppear {
                    appModel.immersiveSpaceState = .open
                    Task { @MainActor in
                        await appModel.prepareToPlay()
                        appModel.media.player.play()
                    }
                }
                .onDisappear {
                }
        }
        .immersionStyle(selection: .constant(.mixed), in: .mixed)
    }
}
