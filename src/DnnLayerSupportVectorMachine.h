// --------------------------------------------------------------------
//  DnnLayerSupportVectorMachine.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerSupportVectorMachine_h
#define DnnLayerSupportVectorMachine_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerSupportVectorMachine : public DnnLayer {
    protected:
    public:
        DnnLayerSupportVectorMachine( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerSupportVectorMachine( void );
        
        virtual bool runForward(  void );
        virtual bool runBackprop( void );
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerSupportVectorMachine_h */
