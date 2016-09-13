// --------------------------------------------------------------------
//  main.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include <ctime>
#include <iostream>
#include "testBuilder.hpp"
#include "testCifar10.hpp"
#include "testFullyConnected.hpp"
#include "TestMatrix.hpp"
#include "test2Layer.hpp"
#include "test2D.hpp"
#include "testIO.hpp"




int
main( int argc, const char * argv[] ) {
    std::cout << "Test DNN begin\n";
    
    const time_t startTime = time( 0 ); 
    
    testMatrix();
    testBuilder();
    testFullyConnected();
    testCifar10();
    testSimple();
    testCircle();
    testSpiral();           // testSpiral may take a a few minutes, depending on your computer.
    test2Layer();
    testIO();
    
    const time_t endTime     = time( 0 );
    const time_t elapsedTime = endTime - startTime;
    
    std::cout << "Test DNN end. Elapsed time: ";
    if( elapsedTime == 1 ) {
       std::cout << "1 second.\n";
    } else {
        std::cout << elapsedTime << " seconds.\n";  // From my 2009 MacBook pro: Elapsed time: 15 seconds.
    }
    return 0;
    
}


