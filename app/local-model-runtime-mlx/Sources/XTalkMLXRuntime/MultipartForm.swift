import Foundation

struct MultipartPart: Sendable {
    let name: String
    let filename: String?
    let body: Data
}

enum MultipartFormError: Error, LocalizedError {
    case missingBoundary
    case invalidBody

    var errorDescription: String? {
        switch self {
        case .missingBoundary:
            "multipart/form-data request has no boundary"
        case .invalidBody:
            "multipart/form-data request is malformed"
        }
    }
}

func parseMultipartForm(contentType: String, body: Data) throws -> [MultipartPart] {
    guard let boundary = contentType
        .split(separator: ";")
        .map({ $0.trimmingCharacters(in: .whitespaces) })
        .first(where: { $0.lowercased().hasPrefix("boundary=") })?
        .dropFirst("boundary=".count),
        !boundary.isEmpty
    else {
        throw MultipartFormError.missingBoundary
    }
    let unquotedBoundary = boundary.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
    let delimiter = Data("--\(unquotedBoundary)".utf8)
    let headerSeparator = Data("\r\n\r\n".utf8)
    let chunks = body.split(separator: delimiter)
    var parts: [MultipartPart] = []

    for rawChunk in chunks {
        var chunk = rawChunk
        if chunk.starts(with: Data("--".utf8)) {
            continue
        }
        if chunk.starts(with: Data("\r\n".utf8)) {
            chunk.removeFirst(2)
        }
        if chunk.suffix(2) == Data("\r\n".utf8) {
            chunk.removeLast(2)
        }
        guard !chunk.isEmpty,
              let headerRange = chunk.range(of: headerSeparator)
        else {
            continue
        }
        let headerData = chunk[..<headerRange.lowerBound]
        guard let headers = String(data: headerData, encoding: .utf8) else {
            throw MultipartFormError.invalidBody
        }
        let contentStart = headerRange.upperBound
        let content = Data(chunk[contentStart...])
        guard let disposition = headers
            .components(separatedBy: "\r\n")
            .first(where: { $0.lowercased().hasPrefix("content-disposition:") }),
            let name = dispositionParameter("name", in: disposition)
        else {
            throw MultipartFormError.invalidBody
        }
        parts.append(
            MultipartPart(
                name: name,
                filename: dispositionParameter("filename", in: disposition),
                body: content
            )
        )
    }
    return parts
}

private func dispositionParameter(_ name: String, in header: String) -> String? {
    let marker = "\(name)=\""
    guard let start = header.range(of: marker) else {
        return nil
    }
    let valueStart = start.upperBound
    guard let end = header[valueStart...].firstIndex(of: "\"") else {
        return nil
    }
    return String(header[valueStart ..< end])
}

private extension Data {
    func split(separator: Data) -> [Data] {
        guard !separator.isEmpty else {
            return [self]
        }
        var result: [Data] = []
        var start = startIndex
        while let range = self.range(
            of: separator,
            options: [],
            in: start ..< endIndex
        ) {
            result.append(Data(self[start ..< range.lowerBound]))
            start = range.upperBound
        }
        result.append(Data(self[start ..< endIndex]))
        return result
    }
}
