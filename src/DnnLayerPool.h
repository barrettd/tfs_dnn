// --------------------------------------------------------------------
//  DnnLayerPool.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerPool_h
#define DnnLayerPool_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerPool : public DnnLayer {
    protected:
    public:
        static const char *className( void );

        DnnLayerPool( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerPool( void );
        
        virtual bool runForward(  void );
        virtual bool runBackprop( void );
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerPool_h */
