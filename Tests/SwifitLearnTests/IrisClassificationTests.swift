
import XCTest
@testable import SwifitLearn

class IrisClassificationTests: XCTestCase {

    func testRandomForest() {
        
        let classifier = RandomForestClassifier(numEstimators: 100, maxDepth: 5, criterion: .gini)
        
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
        let classifier = KNNClassifier(k: 1)
        
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

}
