
import NDArray

class KNeighborClassifier {
    
    let k: Int
    
    var numFeatures: Int?
    var sampleX: NDArray?
    var sampleY: [Int]?
    
    public init(k: Int = 5) {
        precondition(k > 0)
        self.k = k
    }
    
    public func fit(x: NDArray, y: [Int]) {
        precondition(x.shape[0] == y.count)
        precondition(x.ndim == 2)
        
        let numFeatures = x.shape[1]
        self.numFeatures = numFeatures

        self.sampleX = x
        self.sampleY = y
        
    }
    
    public func predict(x: NDArray) -> [Int] {
        precondition(x.ndim == 2)
        precondition(x.shape[1] == numFeatures)
        
        guard let sampleX = self.sampleX, let sampleY = self.sampleY else {
            fatalError("Not fitted.")
        }
        
        return x.map { x in
            let distances = norm(sampleX - x, along: 1)
            let indices = argsort(distances).prefix(k)
            let answers = indices.map { sampleY[$0] }
            return answers.mode()
        }
    }
}
