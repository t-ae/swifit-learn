// swift-tools-version:3.1

import PackageDescription

let package = Package(
    name: "SwifitLearn",
    dependencies: [
        .Package(url: "https://github.com/t-ae/ndarray.git", versions: Version(0, 0, 9)..<Version(1, 0, 0))
    ]
)
