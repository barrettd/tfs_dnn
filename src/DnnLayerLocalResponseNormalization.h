// --------------------------------------------------------------------
//  DnnLayerLocalResponseNormalization.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerLocalResponseNormalization_h
#define DnnLayerLocalResponseNormalization_h


#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerLocalResponseNormalization : public DnnLayer {
    protected:
        
    public:
        static const char *className( void );

        DnnLayerLocalResponseNormalization( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerLocalResponseNormalization( void );
        
        virtual bool runForward(  void );
        virtual bool runBackprop( void );
    };
    
    
}   // namespace tfs



#endif /* DnnLayerLocalResponseNormalization_h */
