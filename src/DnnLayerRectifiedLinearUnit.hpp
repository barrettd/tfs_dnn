// --------------------------------------------------------------------
//  DnnLayerRectifiedLinearUnit.hpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerRectifiedLinearUnit_hpp
#define DnnLayerRectifiedLinearUnit_hpp

#include "DnnLayer.hpp"

namespace tfs {
    
    class DnnLayerRectifiedLinearUnit : public DnnLayer {
    protected:
 
    public:
        DnnLayerRectifiedLinearUnit( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerRectifiedLinearUnit( void );
        
        virtual bool runForward(  void );   // Forward propagate
        virtual bool runBackprop( void );   // Back propagate
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerRectifiedLinearUnit_hpp */
