// --------------------------------------------------------------------
//  Constants.h
//  TFS DNN library level constants are defined here.
//
//  Created by Barrett Davis on 5/10/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef Constants_h
#define Constants_h

namespace tfs {     // Tree Frog Software
    
    // We typically use double for all of our data representations.
    // It is fast with most "big" CPUs.
    // Embedded systems may require different native data types.
    typedef double DNN_NUMERIC;
    typedef long   DNN_INTEGER;
    
}   // namespace tfs


#endif /* Constants_h */
