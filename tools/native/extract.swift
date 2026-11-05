// extract.swift - fronk-extract: text from PDFs and images with macOS's own
// frameworks (PDFKit, Vision), no downloads. Built by `make fronk-extract` into
// runs/bin/fronk-extract and run by tools/fronk_files.py. Prints one JSON object.
//   fronk-extract pdf FILE    {"pages": [text...], "ocr": [page numbers read by OCR]}
//   fronk-extract image FILE  {"width": w, "height": h, "pieces": [{t, x, y, w, h}...]} (boxes 0..1, top-left)
//   fronk-extract warm        loads Vision's text models (the console runs it at start)
// A PDF page with almost no text layer (a scan) is rendered and read by OCR.
import Foundation
import PDFKit
import Vision
import ImageIO
import AppKit

struct Piece { let text: String; let box: CGRect }

/// Recognized text pieces with their boxes (normalized, origin top-left).
func pieces(_ image: CGImage) -> [Piece] {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.automaticallyDetectsLanguage = true
    try? VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    return (request.results ?? []).compactMap { o in
        guard let t = o.topCandidates(1).first?.string else { return nil }
        let b = o.boundingBox
        return Piece(text: t, box: CGRect(x: b.minX, y: 1 - b.maxY, width: b.width, height: b.height))
    }
}

/// Lines top to bottom, then left to right (for scanned PDF pages).
func recognize(_ image: CGImage) -> [String] {
    pieces(image).sorted { a, b in abs(a.box.midY - b.box.midY) > 0.012 ? a.box.midY < b.box.midY : a.box.minX < b.box.minX }.map(\.text)
}

func emit(_ object: Any) {
    let data = try! JSONSerialization.data(withJSONObject: object)
    FileHandle.standardOutput.write(data)
}

let args = CommandLine.arguments
guard args.count == 3 || (args.count == 2 && args[1] == "warm") else {
    FileHandle.standardError.write("usage: fronk-extract pdf|image FILE\n".data(using: .utf8)!); exit(2)
}
let url = URL(fileURLWithPath: args.count == 3 ? args[2] : "/")
switch args[1] {
case "pdf":
    guard let doc = PDFDocument(url: url) else { FileHandle.standardError.write("cannot open PDF\n".data(using: .utf8)!); exit(1) }
    var pages: [String] = [], ocr: [Int] = []
    for i in 0..<doc.pageCount {
        guard let page = doc.page(at: i) else { pages.append(""); continue }
        var text = page.string ?? ""
        if text.trimmingCharacters(in: .whitespacesAndNewlines).count < 25 {
            let box = page.bounds(for: .mediaBox)
            let scale = min(3.0, 2400.0 / max(box.width, box.height))
            let picture = page.thumbnail(of: NSSize(width: box.width * scale, height: box.height * scale), for: .mediaBox)
            if let cg = picture.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                let lines = recognize(cg)
                if !lines.isEmpty { text = lines.joined(separator: "\n"); ocr.append(i + 1) }
            }
        }
        pages.append(text)
    }
    emit(["pages": pages, "ocr": ocr])
case "image":
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { FileHandle.standardError.write("cannot open image\n".data(using: .utf8)!); exit(1) }
    let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] ?? [:]
    let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                    kCGImageSourceCreateThumbnailWithTransform: true,  // EXIF orientation
                                    kCGImageSourceThumbnailMaxPixelSize: 4096]
    guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { exit(1) }
    let found = pieces(image)
    emit(["width": props[kCGImagePropertyPixelWidth] as? Int ?? image.width,
          "height": props[kCGImagePropertyPixelHeight] as? Int ?? image.height,
          "pieces": found.map { ["t": $0.text, "x": $0.box.minX, "y": $0.box.minY, "w": $0.box.width, "h": $0.box.height] }])
case "warm":  // loads Vision's text models so the first real photo is fast
    let ctx = CGContext(data: nil, width: 64, height: 32, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!
    ctx.setFillColor(gray: 1, alpha: 1); ctx.fill(CGRect(x: 0, y: 0, width: 64, height: 32))
    _ = pieces(ctx.makeImage()!)
    emit(["warm": true])
default:
    exit(2)
}
