// --------------------------------------------------------------------
//  DnnLayerSigmoid.hpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerSigmoid_hpp
#define DnnLayerSigmoid_hpp

#include "DnnLayer.hpp"

namespace tfs {
    
    class DnnLayerSigmoid : public DnnLayer {
    protected:
    public:
        DnnLayerSigmoid( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerSigmoid( void );
        
        virtual bool runForward(  void );
        virtual bool runBackprop( void );
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerSigmoid_hpp */
