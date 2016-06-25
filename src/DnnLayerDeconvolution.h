//
//  DnnLayerDeconvolution.hpp
//
//  Created by Barrett Davis on 6/24/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//

#ifndef DnnLayerDeconvolution_h
#define DnnLayerDeconvolution_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerDeconvolution : public DnnLayer {
    protected:
    public:
        DnnLayerDeconvolution( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerDeconvolution( void );
        
        virtual bool runForward(  void );
        virtual bool runBackprop( void );
        
    };
    
    
}   // namespace tfs



#endif /* DnnLayerDeconvolution_h */
