// --------------------------------------------------------------------
//  DnnLayerDropout.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerDropout_h
#define DnnLayerDropout_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerDropout : public DnnLayer {
    protected:
        
    public:
        DnnLayerDropout( void );
        virtual ~DnnLayerDropout( void );
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerDropout_h */
