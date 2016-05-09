// --------------------------------------------------------------------
//  DnnLayerFullyConnected.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerFullyConnected_h
#define DnnLayerFullyConnected_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerFullyConnected : public DnnLayer {
    protected:
        
    public:
        DnnLayerFullyConnected( void );
        DnnLayerFullyConnected( unsigned long xx, unsigned long yy, unsigned long zz = 1 );
        virtual ~DnnLayerFullyConnected( void );
        
    };
    
    
}   // namespace tfs


#endif /* DnnLayerFullyConnected_h */
