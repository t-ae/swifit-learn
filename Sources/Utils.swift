
import Foundation
import NDArray

func uniform(_ upperBound: Int) -> Int {
    return uniform(low: 0, high: upperBound)
}

func uniform(low: Int, high: Int) -> Int {
    precondition(low < high)
    return Int(arc4random_uniform(UInt32(high - low))) + low
}

func argsort(_ arg: NDArray) -> [Int] {
    precondition(arg.ndim == 1)
    return arg.enumerated()
        .sorted { l, r in l.element.asScalar() < r.element.asScalar() }
        .map { $0.offset }
}

func shuffle(_ arg: NDArray) -> NDArray {
    let indices = (0..<arg.shape[0]).map { $0 }
    return arg.select(indices.shuffled())
}

func bootstrapSampling(x: NDArray, y: [Int], numSets: Int) -> [(x: NDArray, y: [Int])] {
    let numSamples = x.shape[0]
    
    let sampleSize = 2*numSamples / 3
    
    var ret: [(x: NDArray, y: [Int])] = []
    
    for _ in 0..<numSets {
        let indices = (0..<sampleSize).map { _ in uniform(numSamples) }
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
    
    func shuffled() -> [Int] {
        var x = self
        for i in 0..<x.count {
            let index = uniform(low: i, high: x.count)
            let tmp = x[index]
            x[index] = x[i]
            x[i] = tmp
        }
        return x
    }
}
