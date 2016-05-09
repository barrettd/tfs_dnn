// --------------------------------------------------------------------
//  DnnLayerSoftmax.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerSoftmax_h
#define DnnLayerSoftmax_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerSoftmax : public DnnLayer {
    protected:
        
    public:
        DnnLayerSoftmax( void );
        virtual ~DnnLayerSoftmax( void );
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerSoftmax_h */
