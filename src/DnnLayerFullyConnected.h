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
        static const char *className( void );

        DnnLayerFullyConnected( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerFullyConnected( void );
        
    };
    
    
}   // namespace tfs


#endif /* DnnLayerFullyConnected_h */
