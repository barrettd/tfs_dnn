// --------------------------------------------------------------------
//  main.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include <iostream>
#include "testCifar10.hpp"
#include "testFullyConnected.hpp"
#include "TestMatrix.hpp"
#include "testSpiral.hpp"




int
main( int argc, const char * argv[] ) {
    std::cout << "Test DNN begin\n";
    
    testMatrix();
    testFullyConnected();
    testCifar10();
//    testSpiral();
    
    std::cout << "Test DNN end\n";
    return 0;
    
}


