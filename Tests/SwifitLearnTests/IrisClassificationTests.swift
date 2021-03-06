
import XCTest
@testable import SwifitLearn

class IrisClassificationTests: XCTestCase {

    func testRandomForest() {
        
        let classifier = RandomForestClassifier(numEstimators: 10, maxDepth: 3, criterion: .gini)
        
        classifier.fit(x: Iris.x_train,
                       y: Iris.y_train.elements().map { Int($0) })
        
        do {
            print("train")
            var correct = 0
            let ys = classifier.predict(x: Iris.x_train)
            for (a, b) in zip(ys, Iris.y_train) {
//                print(a, b.asScalar())
                if a == Int(b.asScalar()) {
                    correct += 1
                }
            }
            print("accuracy: \(Float(correct) / Float(Iris.y_train.shape[0])) (\(correct)/\(Iris.y_train.shape[0]))")
        }
        do {
            print("test")
            var correct = 0
            let ys = classifier.predict(x: Iris.x_test)
            for (a, b) in zip(ys, Iris.y_test) {
//                print(a, b.asScalar())
                if a == Int(b.asScalar()) {
                    correct += 1
                }
            }
            print("accuracy: \(Float(correct) / Float(Iris.y_test.shape[0])) (\(correct)/\(Iris.y_test.shape[0]))")
        }
    }
    
    func testKNN() {
        let classifier = KNeighborClassifier(k: 4)
        
        classifier.fit(x: Iris.x_train,
                       y: Iris.y_train.elements().map { Int($0) })
        
        do {
            print("train")
            var correct = 0
            let ys = classifier.predict(x: Iris.x_train)
            for (a, b) in zip(ys, Iris.y_train) {
                //                print(a, b.asScalar())
                if a == Int(b.asScalar()) {
                    correct += 1
                }
            }
            print("accuracy: \(Float(correct) / Float(Iris.y_train.shape[0])) (\(correct)/\(Iris.y_train.shape[0]))")
        }
        do {
            print("test")
            var correct = 0
            let ys = classifier.predict(x: Iris.x_test)
            for (a, b) in zip(ys, Iris.y_test) {
                //                print(a, b.asScalar())
                if a == Int(b.asScalar()) {
                    correct += 1
                }
            }
            print("accuracy: \(Float(correct) / Float(Iris.y_test.shape[0])) (\(correct)/\(Iris.y_test.shape[0]))")
        }
    }
    
    func testKMeans() {
        let km = KMeans(k: 3, numSteps: 30)
        
        let ys = km.fit(x: Iris.x_train)
        
        var clusterToClass = [Int: Int]()
        
        for i in 0..<3 {
            let bucket = zip(ys, Iris.y_train).flatMap { $0.0 == i ? Int($0.1.asScalar()) : nil }
            let mode = bucket.mode()
            print("Cluster \(i), mode: \(mode)")
            clusterToClass[i] = mode
        }
        
        do {
            var correct = 0
            for (a, b) in zip(ys, Iris.y_train) {
                let a = clusterToClass[a]!
                // print(a, b.asScalar())
                if a == Int(b.asScalar()) {
                    correct += 1
                }
            }
            print("accuracy: \(Float(correct) / Float(Iris.y_train.shape[0])) (\(correct)/\(Iris.y_train.shape[0]))")
        }
        do {
            var correct = 0
            let ys = km.predict(x: Iris.x_test)
            for (a, b) in zip(ys, Iris.y_test) {
                let a = clusterToClass[a]!
                // print(a, b.asScalar())
                if a == Int(b.asScalar()) {
                    correct += 1
                }
            }
            print("accuracy: \(Float(correct) / Float(Iris.y_test.shape[0])) (\(correct)/\(Iris.y_test.shape[0]))")
        }
    }
    
    func testMeanShift() {
        let ms = MeanShift(bandwidth: 0.86, clusterAll: true)
        
        let ys = ms.fit(x: Iris.x_train)
        print("numClusters: \(ms.numClusters!)")
        
        var clusterToClass = [Int: Int]()
        
        for i in 0..<3 {
            let bucket = zip(ys, Iris.y_train).flatMap { $0.0 == i ? Int($0.1.asScalar()) : nil }
            let mode = bucket.mode()
            print("Cluster \(i), mode: \(mode)")
            clusterToClass[i] = mode
        }
        
        do {
            var correct = 0
            var missing = 0
            for (a, b) in zip(ys, Iris.y_train) {
                guard let a = clusterToClass[a] else {
                    missing += 1
                    continue
                }
                // print(a, b.asScalar())
                if a == Int(b.asScalar()) {
                    correct += 1
                }
            }
            print("accuracy: \(Float(correct) / Float(Iris.y_train.shape[0])) (\(correct)/\(Iris.y_train.shape[0]))")
            print("missing: \(missing)")
        }
        do {
            let ys = ms.predict(x: Iris.x_test)
            var correct = 0
            var missing = 0
            for (a, b) in zip(ys, Iris.y_test) {
                guard let a = clusterToClass[a] else {
                    missing += 1
                    continue
                }
                // print(a, b.asScalar())
                if a == Int(b.asScalar()) {
                    correct += 1
                }
            }
            print("accuracy: \(Float(correct) / Float(Iris.y_test.shape[0])) (\(correct)/\(Iris.y_test.shape[0]))")
            print("missing: \(missing)")
        }
    }

    func testDBSCAN() {
        let db = DBSCAN(eps: 0.6)
        
        let ys = db.fit(x: Iris.x_train)
        print("numClusters: \(db.numClusters!)")
        
        var clusterToClass = [Int: Int]()
        
        for i in 0..<3 {
            let bucket = zip(ys, Iris.y_train).flatMap { $0.0 == i ? Int($0.1.asScalar()) : nil }
            let mode = bucket.mode()
            print("Cluster \(i), mode: \(mode)")
            clusterToClass[i] = mode
        }
        
        do {
            var correct = 0
            var missing = 0
            for (a, b) in zip(ys, Iris.y_train) {
                guard let a = clusterToClass[a] else {
                    missing += 1
                    continue
                }
                // print(a, b.asScalar())
                if a == Int(b.asScalar()) {
                    correct += 1
                }
            }
            print("accuracy: \(Float(correct) / Float(Iris.y_train.shape[0])) (\(correct)/\(Iris.y_train.shape[0]))")
            print("missing: \(missing)")
        }
        do {
            let ys = db.predict(x: Iris.x_test)
            var correct = 0
            var missing = 0
            for (a, b) in zip(ys, Iris.y_test) {
                guard let a = clusterToClass[a] else {
                    missing += 1
                    continue
                }
                // print(a, b.asScalar())
                if a == Int(b.asScalar()) {
                    correct += 1
                }
            }
            print("accuracy: \(Float(correct) / Float(Iris.y_test.shape[0])) (\(correct)/\(Iris.y_test.shape[0]))")
            print("missing: \(missing)")
        }
    }
}
