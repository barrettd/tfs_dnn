// --------------------------------------------------------------------
//  DnnLayerTanh.hpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerTanh_hpp
#define DnnLayerTanh_hpp

#include "DnnLayer.hpp"

namespace tfs {
    
    class DnnLayerTanh : public DnnLayer {      // tanh( radians ) output is between [-1.0, 1.0]
    protected:
    public:
        DnnLayerTanh( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerTanh( void );
        
        virtual bool runForward(  void );
        virtual bool runBackprop( void );
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerTanh_hpp */
