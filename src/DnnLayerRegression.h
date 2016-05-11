// --------------------------------------------------------------------
//  DnnLayerRegression.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerRegression_h
#define DnnLayerRegression_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerRegression : public DnnLayer {
    protected:
        
    public:
        static const char *className( void );

        DnnLayerRegression( DnnLayer *previousLayer );
        virtual ~DnnLayerRegression( void );
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerRegression_h */
