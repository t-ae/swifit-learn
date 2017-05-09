
import Foundation
import NDArray

func bootstrapSampling(x: NDArray, y: [Int], numSets: Int) -> [(x: NDArray, y: [Int])] {
    let numSamples = x.shape[0]
    
    let sampleSize = 2*numSamples / 3
    
    var ret: [(x: NDArray, y: [Int])] = []
    
    for _ in 0..<numSets {
        let indices = (0..<sampleSize).map { _ in Int(arc4random_uniform(UInt32(numSamples))) }
        ret.append((x: NDArray.stack(indices.map { x[$0] }),
                    y: indices.map { y[$0] }))
    }
    
    return ret
}

extension Array where Element == Int {
    func mode() -> Int {
        var dict = [Int: Int]()
        for e in self {
            if dict[e] == nil {
                dict[e] = 0
            }
            dict[e]! += 1
        }
        var max = 0
        var maxElement: Int? = nil
        for (k, v) in dict {
            if v > max {
                max = v
                maxElement = k
            }
        }
        return maxElement!
    }
}
