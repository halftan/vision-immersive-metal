//
//  ContentView.swift
//  vision-immersive-metal
//
//

import SwiftUI
import RealityKit
import RealityKitContent
internal import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(AppModel.self) private var appModel
    @State private var showFileImporter = false

    var body: some View {
        VStack {
            Button("Select video", systemImage: "arrow.forward.folder.fill") {
                showFileImporter = true
            }
            .fileImporter(isPresented: $showFileImporter, allowedContentTypes: [.movie, .video]) { result in
                switch result {
                case .success(let url):
                    appModel.videoURL = url
                    print("Selected file: \(url)")
                case .failure(let error):
                    print("Failed to open file: \(error.localizedDescription)")
                }
            }
            ToggleImmersiveSpaceButton()
            
            Text("Selected file: \(appModel.videoURL?.absoluteString ?? "None")")
                .padding()
                .background(Color.secondary.opacity(0.2))
                .cornerRadius(10)
                .padding(.top)
            
            Spacer()
            
            TransportControlsView(appModel.media)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16.0, style: .continuous))
        }
        .padding()
    }
}

#Preview(windowStyle: .automatic) {
    ContentView()
        .environment(AppModel())
}
