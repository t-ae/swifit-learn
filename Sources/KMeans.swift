
import Foundation
import NDArray

public class KMeans {
    
    let k: Int
    let numSteps: Int
    
    public internal(set) var centers: NDArray?
    
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
        
        let xShuffle = shuffle(x)
        for i in 0..<k {
            bucket[i] = xShuffle[i...~>k]
        }
        var centers = NDArray.stack(bucket.values.map { mean($0, along: 0) })
        var lastIndices = NDArray.empty([0])
        
        for _ in 0..<numSteps {
            let ds = norm(centers.expandDims(0) - x.expandDims(1), along: -1)
            let indices = argmin(ds, along: 1)
            
            guard lastIndices != indices else {
                break
            }
            for i in 0..<k {
                let mask = indices.map { Int($0.asScalar()) == i }
                bucket[i] = x.select(mask)
            }
            centers = NDArray.stack(bucket.values.map { mean($0, along: 0) })
            lastIndices = indices
        }
        self.centers = centers
        return predict(x: x)
    }
    
    public func predict(x: NDArray) -> [Int] {
        precondition(x.ndim == 2)
        guard let centers = self.centers else {
            fatalError("Not fitted.")
        }
        
        let ds = norm(centers.expandDims(0) - x.expandDims(1), along: -1)
        let indices = argmin(ds, along: 1)
        
        return indices.map { Int($0.asScalar()) }
    }
}
