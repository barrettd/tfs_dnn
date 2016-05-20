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

    public:
        static const char *className( void );

        DnnLayerSoftmax( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerSoftmax( void );
      
        virtual bool forward(  void );                  // Forward propagate while training
         virtual DNN_NUMERIC backprop( const DMatrix &expectation );  // Last layer
        
        virtual bool predict( const Matrix &data );     // Forward progagate when predicting
    };
    
    
}   // namespace tfs

#endif /* DnnLayerSoftmax_h */
