//
//  UtilsTests.swift
//  SwifitLearn
//
//  Created by Araki Takehiro on 2017/05/16.
//
//

import XCTest
@testable import SwifitLearn

class UtilsTests: XCTestCase {

    func testShuffle() {
        let a = (0..<10).map { $0 }
        let shuffle = a.shuffled()
        print(shuffle)
        
        XCTAssertEqual(Set(shuffle), Set(a))
        XCTAssertEqual(Set(shuffle).count, a.count)
    }

}
