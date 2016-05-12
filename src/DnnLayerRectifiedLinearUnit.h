// --------------------------------------------------------------------
//  DnnLayerRectifiedLinearUnit.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerRectifiedLinearUnit_h
#define DnnLayerRectifiedLinearUnit_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerRectifiedLinearUnit : public DnnLayer {
    protected:
        bool threshold( const Matrix &data );
        
    public:
        static const char *className( void );

        DnnLayerRectifiedLinearUnit( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerRectifiedLinearUnit( void );
        
        virtual bool forward( const Matrix &data );     // Forward propagate while training
        virtual bool backprop( void );                  // Back propagate while training
        virtual bool predict( const Matrix &data );     // Forward progagate when predicting

    };
    
    
}   // namespace tfs

#endif /* DnnLayerRectifiedLinearUnit_h */
