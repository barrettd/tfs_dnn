// --------------------------------------------------------------------
//  DnnLayerSoftmax.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerSoftmax_h
#define DnnLayerSoftmax_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerSoftmax : public DnnLayer {
    protected:
        Matrix *m_es;       // Exponentials
        
        void setup( const bool trainable = true );

        virtual bool runForward(  void );   // Forward propagate
        virtual DNN_NUMERIC runBackprop( const DNN_INTEGER expectation );   // Back propagate

    public:
        static const char *className( void );

        DnnLayerSoftmax( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerSoftmax( void );
      
    };
    
    
}   // namespace tfs

#endif /* DnnLayerSoftmax_h */
