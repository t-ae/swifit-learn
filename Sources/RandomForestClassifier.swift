
import Foundation
import NDArray

public enum Criterion {
    case gini, entropy
}

public enum MaxFeatures {
    case auto
    case int(Int)
}

public class RandomForestClassifier {
    
    let numEstimators: Int
    let maxDepth: Int
    let criterion: Criterion
    let maxFeatures: MaxFeatures
    
    var numFeatures: Int?
    
    var trees: [Node]?
    
    public init(numEstimators: Int = 10,
                maxDepth: Int = 2,
                criterion: Criterion = .gini,
                maxFeatures: MaxFeatures = .auto) {
        self.numEstimators = numEstimators
        self.maxDepth = maxDepth
        self.criterion = criterion
        self.maxFeatures = maxFeatures
    }
    
    public func fit(x: NDArray, y: [Int]) {
        precondition(x.shape[0] == y.count)
        precondition(x.ndim == 2)
        
        let numFeatures = x.shape[1]
        self.numFeatures = numFeatures
        
        let samples = bootstrapSampling(x: x, y: y, numSets: numEstimators)
        
        let maxFeatures: Int
        switch self.maxFeatures {
        case .auto:
            maxFeatures = Int(sqrt(Double(numFeatures)))
        case let .int(n):
            precondition(n > 0)
            maxFeatures = min(numFeatures, n)
        }
        
        self.trees = (0..<numEstimators).map { i in Node(x: samples[i].x, y: samples[i].y, criterion: criterion) }
        DispatchQueue.concurrentPerform(iterations: numEstimators) { i in
            self.trees![i].divide(maxDepth: maxDepth, maxFeatures: maxFeatures)
        }
    }
    
    public func predict(x: NDArray) -> [Int] {
        precondition(x.ndim == 2)
        precondition(x.shape[1] == numFeatures)
        guard let trees = self.trees else {
            fatalError("Not fitted")
        }
        return x.map { x in
            
            let bin = bincount(trees.map { $0.predict(x: x) })
            var max = -1
            var ans = -1
            for (k, v) in bin {
                if v > max {
                    max = v
                    ans = k
                }
            }
            return ans
        }
    }
}

class Node {
    let x: NDArray
    let y: [Int]
    let criterion: Criterion
    
    // leaf
    var answer: Int?
    
    // node
    var divisionFeature: Int?
    var divisionBoundary: Float?
    var left: Node?
    var right: Node?
    
    init(x: NDArray, y: [Int], criterion: Criterion) {
        self.x = x
        self.y = y
        self.criterion = criterion
    }
    
    func divide(maxDepth: Int, maxFeatures: Int) {
        guard y.contains(where: { $0 != y[0] }) else {
            answer = y[0]
            return
        }
        guard maxDepth > 0 else {
            answer = y.mode()
            return
        }
        
        let featureCandidates = [Int](0..<x.shape[1]).shuffled().prefix(maxFeatures)
        
        var minScore = Float.infinity
        var divisionFeature = -1
        var divisionBoundary = Float.nan
        for feature in featureCandidates {
            let boundaryCandidates = x[nil, feature].elements()
            
            for bound in boundaryCandidates {
                let indices = x[nil, feature].indices { $0.asScalar() < bound }
                let lbin = bincount(indices.map { y[$0] })
                let rbin = bincount(Set(0..<y.count).subtracting(Set(indices)).map { y[$0] })
                
                if lbin.isEmpty || rbin.isEmpty {
                    continue
                }
                let score = (calcScore(lbin)*Float(lbin.count) + calcScore(rbin)*Float(rbin.count)) / Float(lbin.count+rbin.count)
                if minScore > score {
                    minScore = score
                    divisionFeature = feature
                    divisionBoundary = bound
                }
            }
        }
        
        guard divisionFeature != -1 else {
            // every sample has same value
            answer = y.mode()
            return
        }
        self.divisionFeature = divisionFeature
        self.divisionBoundary = divisionBoundary
        
        let ls = x.enumerated().filter { $1[divisionFeature].asScalar() < divisionBoundary }.map { $0.offset }
        let rs = [Int](Set(0..<x.shape[0]).subtracting(Set(ls)))
        
        let lx = x.select(ls)
        let ly = ls.map { y[$0] }
        let rx = x.select(rs)
        let ry = rs.map { y[$0] }
        
        left = Node(x: lx, y: ly, criterion: criterion)
        right = Node(x: rx, y: ry, criterion: criterion)
        
        left!.divide(maxDepth: maxDepth - 1, maxFeatures: maxFeatures)
        right!.divide(maxDepth: maxDepth - 1, maxFeatures: maxFeatures)
    }
    
    func calcScore(_ bin: [Int: Int]) -> Float {
        let p = NDArray(bin.values.map { Float($0) }) / Float(bin.values.reduce(0, +))

        switch criterion {
        case .gini:
            return 1 - sum(p*p).asScalar()
        case .entropy:
            return -sum(p*log(p)).asScalar()
        }
    }
    
    func predict(x: NDArray) -> Int {
        precondition(x.ndim == 1)
        
        if let answer = self.answer {
            return answer
        } else if x[divisionFeature!].asScalar() < divisionBoundary! {
            return left!.predict(x: x)
        } else {
            return right!.predict(x: x)
        }
    }
}
