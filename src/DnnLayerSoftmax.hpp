// --------------------------------------------------------------------
//  DnnLayerSoftmax.hpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerSoftmax_hpp
#define DnnLayerSoftmax_hpp

#include "DnnLayer.hpp"

namespace tfs {
    
    class DnnLayerSoftmax : public DnnLayer {
    protected:
        Matrix *m_es;       // Exponentials
        
        void setup( const bool trainable = true );

    public:
        DnnLayerSoftmax( DnnLayer *previousLayer, const bool trainable = true );
        virtual ~DnnLayerSoftmax( void );
        
        Matrix *exponentialsReadOnly( void ) const;
        Matrix *exponentials( void );
      
        virtual bool runForward(  void );   // Forward propagate
        virtual DNN_NUMERIC runBackprop( const DNN_INTEGER expectation );   // Back propagate
        
        DNN_INTEGER getPrediction( void ) const;    // Return index of max probability
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerSoftmax_hpp */
