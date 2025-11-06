//
//  AppModel.swift
//  vision-immersive-metal
//
//

import AVFoundation
import SwiftUI

/// Maintains app-wide state
@MainActor
@Observable
class AppModel {
    let immersiveSpaceID = "ImmersiveSpace"
    enum ImmersiveSpaceState {
        case closed
        case inTransition
        case open
    }
    var immersiveSpaceState = ImmersiveSpaceState.closed

    var videoURL: URL?
    private(set) var asset: AVAsset?

    var media = MediaPlaybackModel()
    var videoOutput: AVPlayerItemVideoOutput?
    
    init() {
        prepareVideoOutput()
    }

    func prepareToPlay() async {
        guard let videoURL = self.videoURL else {
            print("videoURL not set!")
            return
        }
        guard videoURL.startAccessingSecurityScopedResource() else {
            print("Failed to access security-scoped resource at \(videoURL.absoluteString)")
            return
        }
        asset = AVURLAsset(url: videoURL)
        guard let asset = self.asset else {
            print("AVAsset not ready!")
            return
        }
        
        let playerItem = AVPlayerItem(asset: asset)
        playerItem.add(videoOutput!)
        print("Set current playitem")
        media.setCurrentItem(playerItem)
    }
    
    private func prepareVideoOutput() {
        let videoColorProperties = [
            AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_2020,
            AVVideoTransferFunctionKey: AVVideoTransferFunction_Linear,
            AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_2020,
        ]
        let outputVideoSettings: [String: Any] = [
            AVVideoAllowWideColorKey: true,
            AVVideoColorPropertiesKey: videoColorProperties,
            kCVPixelBufferPixelFormatTypeKey as String: NSNumber(
                value: kCVPixelFormatType_64RGBAHalf
            ),
        ]
        
        // Create the video output on the main actor
        let videoOutput = AVPlayerItemVideoOutput(
            outputSettings: outputVideoSettings
        )
        
        print("Generated video output \(videoOutput)")
        self.videoOutput = videoOutput
    }

    func releaseAsset() {
        self.videoURL?.stopAccessingSecurityScopedResource()
    }
}
