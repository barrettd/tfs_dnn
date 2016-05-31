// --------------------------------------------------------------------
//  main.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include <iostream>
#include "testBuilder.hpp"
#include "testCifar10.hpp"
#include "testCircle.hpp"
#include "testFullyConnected.hpp"
#include "TestMatrix.hpp"
#include "testSpiral.hpp"
#include "test2Layer.hpp"




int
main( int argc, const char * argv[] ) {
    std::cout << "Test DNN begin\n";
    
    testMatrix();
    testBuilder();
    testFullyConnected();
    testCifar10();
    testCircle();
    testSpiral();
    test2Layer();
    
    std::cout << "Test DNN end\n";
    return 0;
    
}


