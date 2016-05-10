// --------------------------------------------------------------------
//  DnnLayerSigmoid.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerSigmoid_h
#define DnnLayerSigmoid_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerSigmoid : public DnnLayer {
    protected:
        
    public:
        static const char *className( void );

        DnnLayerSigmoid( void );
        virtual ~DnnLayerSigmoid( void );
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerSigmoid_h */
