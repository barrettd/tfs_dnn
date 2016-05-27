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
        bool threshold( const DNN_NUMERIC *src, const DNN_NUMERIC * const end, DNN_NUMERIC *dst );
 
    public:
        static const char *className( void );

        DnnLayerRectifiedLinearUnit( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerRectifiedLinearUnit( void );
        
        virtual bool runForward(  void );   // Forward propagate
        virtual bool runBackprop( void );   // Back propagate
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerRectifiedLinearUnit_h */
