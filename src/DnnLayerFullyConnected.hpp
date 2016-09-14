// --------------------------------------------------------------------
//  DnnLayerFullyConnected.hpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerFullyConnected_hpp
#define DnnLayerFullyConnected_hpp

#include "DnnLayer.hpp"

namespace tfs {
    
    class DnnLayerFullyConnected : public DnnLayer {
    protected:
        unsigned long m_neuron_count;   // Always contains a bias for each neuron.
        
        void setup( const bool trainable = true );
        
    public:
        DnnLayerFullyConnected( DnnLayer *previousLayer, unsigned long neuronCount, const bool trainable = true );
        virtual ~DnnLayerFullyConnected( void );
        
        unsigned long getNeuronCount( void ) const;

        virtual bool runBias( DNN_NUMERIC value = 0.0 );    // Generally random biases seem to work quite well.
        
        unsigned long neuronCount( void ) const;
        
        virtual bool runForward(  void );   // Forward propagate
        virtual bool runBackprop( void );   // Back propagate
    };
    
    
}   // namespace tfs


#endif /* DnnLayerFullyConnected_hpp */
