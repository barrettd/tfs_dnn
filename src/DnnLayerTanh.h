// --------------------------------------------------------------------
//  DnnLayerTanh.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerTanh_h
#define DnnLayerTanh_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerTanh : public DnnLayer {
    protected:
        
    public:
        static const char *className( void );

        DnnLayerTanh( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerTanh( void );
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerTanh_h */
