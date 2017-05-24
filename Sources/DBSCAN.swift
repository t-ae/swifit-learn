
import NDArray

public class DBSCAN {
    
    let eps: Float
    let minSamples: Int
    
    var corePoints: NDArray?
    var clusters: [Int]?
    public internal(set) var numClusters: Int?
    
    public init(eps: Float = 0.5, minSamples: Int = 5) {
        self.eps = eps
        self.minSamples = minSamples
    }
    
    public func fit(x: NDArray) -> [Int] {
        precondition(x.ndim == 2)
        
        let distances = norm(x.expandDims(0) - x.expandDims(1), along: -1)
        let d = distances - eps + 2*eps*NDArray.eye(x.shape[0])
        
        let matrix = (copySign(magnitude: NDArray(scalar: 1), sign: d) - 1) / -2
        
        var ret = [Int](repeating: -1, count: x.shape[0])
        var cluster = 0
        let coreIndices = sum(matrix, along: -1).indices { Int($0.asScalar()) >= minSamples }
        
        func setCluster(index: Int) {
            guard ret[index] == -1 else {
                return
            }
            ret[index] = cluster
            if coreIndices.contains(index) {
                for i in matrix[index].indices(where: { Int($0.asScalar()) == 1 }) {
                    setCluster(index: i)
                }
            }
        }
        
        for index in coreIndices {
            guard ret[index] == -1 else {
                continue
            }
            setCluster(index: index)
            cluster += 1
        }
        
        corePoints = x.select(coreIndices)
        clusters = coreIndices.map { ret[$0] }
        numClusters = cluster
        
        return ret
    }
    
    public func predict(x: NDArray) -> [Int] {
        precondition(x.ndim == 2)
        guard let corePoints = self.corePoints, let clusters = self.clusters else {
            fatalError("Not fitted.")
        }
        let distances = norm(corePoints.expandDims(0) - x.expandDims(1), along: -1)
        
        return distances.map { ds in
            guard let i = ds.indices(where: { $0.asScalar() <= eps }).first else {
                return -1
            }
            return clusters[i]
        }
    }
    
}
