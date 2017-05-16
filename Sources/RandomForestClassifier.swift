
import Foundation
import NDArray

public enum Criterion {
    case gini, entropy
}

public class RandomForestClassifier {
    
    let numEstimators: Int
    let maxDepth: Int
    let criterion: Criterion
    
    var numFeatures: Int?
    
    var trees: [Node]?
    
    public init(numEstimators: Int = 10, maxDepth: Int = 2, criterion: Criterion = .gini) {
        self.numEstimators = numEstimators
        self.maxDepth = maxDepth
        self.criterion = criterion
    }
    
    public func fit(x: NDArray, y: [Int]) {
        precondition(x.shape[0] == y.count)
        precondition(x.ndim == 2)
        
        let numFeatures = x.shape[1]
        self.numFeatures = numFeatures
        
        let samples = bootstrapSampling(x: x, y: y, numSets: numEstimators)
        
        self.trees = (0..<numEstimators).map { i in Node(x: samples[i].x, y: samples[i].y, criterion: criterion) }
        DispatchQueue.concurrentPerform(iterations: numEstimators) { i in
            self.trees![i].divide(maxDepth: maxDepth)
        }
    }
    
    public func predict(x: NDArray) -> [Int] {
        precondition(x.ndim == 2)
        precondition(x.shape[1] == numFeatures)
        guard let trees = self.trees else {
            fatalError("Not fitted")
        }
        return x.map { x in
            var bins = [Int: Int]()
            for tree in trees {
                let ans = tree.predict(x: x)
                if bins[ans] == nil {
                    bins[ans] = 0
                }
                bins[ans]! += 1
            }
            var max = -1
            var ans = -1
            for (k, v) in bins {
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
    
    func divide(maxDepth: Int) {
        guard y.contains(where: { $0 != y[0] }) else {
            answer = y[0]
            return
        }
        guard maxDepth > 0 else {
            answer = y.mode()
            return
        }
        
        let divisionFeature = uniform(x.shape[1])
        self.divisionFeature = divisionFeature
        
        let candidates = x[nil, divisionFeature].elements()
        
        var minScore = Float.infinity
        var divisionBoundary = Float.nan
        
        for c in candidates {
            var lbin = [Int: Int]()
            var rbin = [Int: Int]()
            for (sample, gt) in zip(x, y) {
                if sample[divisionFeature].asScalar() < c {
                    if lbin[gt] == nil {
                        lbin[gt] = 0
                    }
                    lbin[gt]! += 1
                } else {
                    if rbin[gt] == nil {
                        rbin[gt] = 0
                    }
                    rbin[gt]! += 1
                }
            }
            if lbin.isEmpty || rbin.isEmpty {
                continue
            }
            let score = (calcScore(lbin)*Float(lbin.count) + calcScore(rbin)*Float(rbin.count)) / Float(lbin.count+rbin.count)
            if minScore > score {
                minScore = score
                divisionBoundary = c
            }
        }
        guard !divisionBoundary.isNaN else {
            // every sample has same value
            answer = y.mode()
            return
        }
        self.divisionBoundary = divisionBoundary
        
        let ls = x.enumerated().filter { $1[divisionFeature].asScalar() < divisionBoundary }.map { $0.offset }
        let rs = [Int](Set(0..<x.shape[0]).subtracting(Set(ls)))
        
        let lx = x.select(ls)
        let ly = ls.map { y[$0] }
        let rx = x.select(rs)
        let ry = rs.map { y[$0] }
        
        left = Node(x: lx, y: ly, criterion: criterion)
        right = Node(x: rx, y: ry, criterion: criterion)
        
        left!.divide(maxDepth: maxDepth - 1)
        right!.divide(maxDepth: maxDepth - 1)
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
