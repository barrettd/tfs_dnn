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
    
    class DnnLayerTanh : public DnnLayer {      // tanh( radians ) output is between [-1.0, 1.0]
    public:
        static const char *className( void );

        DnnLayerTanh( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerTanh( void );
        
        virtual bool forward(  void );                  // Forward propagate while training
        virtual bool backprop( void );                  // Back propagate while training
        
        virtual bool predict( const Matrix &data );     // Forward progagate when predicting

    };
    
    
}   // namespace tfs

#endif /* DnnLayerTanh_h */
