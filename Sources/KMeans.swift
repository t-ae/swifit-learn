
import Foundation
import NDArray

public class KMeans {
    
    let k: Int
    let numSteps: Int
    
    var centers: [Int: NDArray]?
    
    public init(k: Int, numSteps: Int = 10) {
        precondition(k > 1)
        precondition(numSteps > 0)
        self.k = k
        self.numSteps = numSteps
    }
    
    public func fit(x: NDArray) -> [Int] {
        precondition(x.ndim == 2)
        precondition(x.shape[0] >= k)
        
        var bucket = [Int: NDArray]()
        var centers = [Int: NDArray]()
        var lastIndices: [Int] = []
        
        let xShuffle = shuffle(x)
        for i in 0..<k {
            bucket[i] = xShuffle[i...~>k]
            centers[i] = mean(bucket[i]!, along: 0)
        }
        
        for _ in 0..<numSteps {
            let indices = x.map { x -> Int in
                var max = Float.infinity
                var maxIndex = -1
                for i in 0..<k {
                    let distance = norm(centers[i]! - x)
                    if distance < max {
                        max = distance
                        maxIndex = i
                    }
                }
                return maxIndex
            }
            guard lastIndices != indices else {
                break
            }
            for i in 0..<k {
                let mask = indices.map { $0 == i }
                bucket[i] = x.select(mask)
                centers[i] = mean(bucket[i]!, along: 0)
            }
            lastIndices = indices
        }
        self.centers = centers
        return lastIndices
    }
    
    public func predict(x: NDArray) -> [Int] {
        
        guard let centers = self.centers else {
            fatalError("Not fitted.")
        }
        
        let indices = x.map { x -> Int in
            var max = Float.infinity
            var maxIndex = -1
            for i in 0..<k {
                let distance = norm(centers[i]! - x)
                if distance < max {
                    max = distance
                    maxIndex = i
                }
            }
            return maxIndex
        }
        return indices
    }
}
