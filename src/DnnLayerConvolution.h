// --------------------------------------------------------------------
//  DnnLayerConvolution.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerConvolution_h
#define DnnLayerConvolution_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerConvolution : public DnnLayer {
    protected:
        
    public:
        static const char *className( void );

        DnnLayerConvolution( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerConvolution( void );
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerConvolution_h */
