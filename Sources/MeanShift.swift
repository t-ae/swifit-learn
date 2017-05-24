
import NDArray

public class MeanShift {
    
    let bandwidth: Float
    let maxIter: Int
    let clusterAll: Bool
    
    public internal(set) var centers: NDArray?
    
    public var numClusters: Int? {
        return centers?.shape[0]
    }
    
    public init(bandwidth: Float, maxIter: Int = 300, clusterAll: Bool = true) {
        precondition(bandwidth > 0)
        precondition(maxIter > 0)
        self.bandwidth = bandwidth
        self.maxIter = maxIter
        self.clusterAll = clusterAll
    }
    
    public func fit(x: NDArray) -> [Int] {
        precondition(x.ndim == 2)
        
        let allCenters = NDArray.stack(x.map { seed -> NDArray in
            var center = seed
            var inCircle = NDArray.empty([0])
            
            for _ in 0..<self.maxIter {
                let newInCircle = x.select(norm(x - center, along: 1).indices { $0.asScalar() <= bandwidth })
                if newInCircle == inCircle {
                    break
                }
                inCircle = newInCircle
                center = mean(inCircle, along: 0)
            }
            return center
        })
        
        let d = norm(allCenters.expandDims(1) - x.expandDims(0), along: -1) //  numcenters x numsamples
        let dd = d - bandwidth
        let counts = sum(dd.clipped(low: 0) / dd, along: 1)
        
        let sortedCenters = allCenters.select(argsort(counts, ascending: false))
        
        var mask = [Bool](repeating: true, count: sortedCenters.shape[0])
        for i in 0..<mask.count-1 {
            let distances = norm(sortedCenters[(i+1)...] - sortedCenters[i], along: 1)
            let _mask = distances.map { $0.asScalar() > bandwidth }
            for j in i+1..<mask.count {
                mask[j] = mask[j] && _mask[j-i-1]
            }
        }
        
        let centers = sortedCenters.select(mask)
        self.centers = centers
        
        
        let distances = norm(centers.expandDims(0) - x.expandDims(1), along: -1)
        if clusterAll {
            return argmin(distances, along: 1).map { Int($0.asScalar()) }
        } else {
            return distances.map { row in
                let min = Int(argmin(row, along: 0).asScalar())
                if row[min].asScalar() > bandwidth {
                    return -1
                } else {
                    return min
                }
            }
        }
    }
    
    public func predict(x: NDArray) -> [Int] {
        precondition(x.ndim == 2)
        guard let centers = self.centers else {
            fatalError("Not fitted.")
        }
        
        let distances = norm(centers.expandDims(0) - x.expandDims(1), along: -1)
        if clusterAll {
            return argmin(distances, along: 1).map { Int($0.asScalar()) }
        } else {
            return distances.map { row in
                let min = Int(argmin(row, along: 0).asScalar())
                if row[min].asScalar() > bandwidth {
                    return -1
                } else {
                    return min
                }
            }
        }
    }
}
