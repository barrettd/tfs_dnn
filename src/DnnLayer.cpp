// --------------------------------------------------------------------
//  DnnLayer.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayer.h"
#include "Error.h"

namespace tfs {
    
    DnnLayer::DnnLayer( const char *name ):
    m_name( name ),
    m_in_a( 0 ), m_in_dw( 0 ), m_w( 0 ), m_dw( 0 ), m_out_a( 0 ), m_out_dw( 0 ),
    m_l1_decay_mul( 1.0 ), m_l2_decay_mul( 1.0 ),
    m_prev_layer( 0 ), m_next_layer( 0 ) {
        // Constructor
    }
    
    DnnLayer::DnnLayer( const char *name, DnnLayer *previousLayer ) :
    m_name( name ),
    m_in_a( 0 ), m_in_dw( 0 ), m_w( 0 ), m_dw( 0 ), m_out_a( 0 ), m_out_dw( 0 ),
    m_l1_decay_mul( 1.0 ), m_l2_decay_mul( 1.0 ),
    m_prev_layer( previousLayer ), m_next_layer( 0 ) {  // Constructor
        if( previousLayer != 0 ) {
            m_in_a  = previousLayer->m_out_a;            // Wire up to previous layer activations as input to forward() and predict()
            m_in_dw = previousLayer->m_out_dw;
            previousLayer->setNextLayer( this );
        } else {
            log_error( "Previous layer is null" );
        }
    }
    
    DnnLayer::~DnnLayer( void ) {
        teardown();
        m_prev_layer = 0;
        m_next_layer = 0;
    }

    void
    DnnLayer::setup( const bool trainable ) {
        // -----------------------------------------------------------------------------------
        // S = size of input data
        // out_a[S]  = activations of each neuron
        // out_dw[S] = gradiant
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 ) {
            log_error( "Input activation matrix is null" );
            return;
        }
        if( trainable && m_in_dw == 0 ) {
            log_error( "Input dw matrix is null" );
            return;
        }
        m_out_a = new Matrix( *m_in_a );         // Output dimension matches input dimension
        if( trainable ) {
            m_out_dw = new Matrix( *m_in_dw );   // Output dimension matches input dimension
        }
        return;
    }

    void
    DnnLayer::teardown( void ) {
        delete m_w;
        delete m_dw;
        delete m_out_a;
        delete m_out_dw;
        m_in_a   = 0;
        m_in_dw  = 0;
        m_w      = 0;
        m_dw     = 0;
        m_out_a  = 0;
        m_out_dw = 0;
        return;
    }

    const char*
    DnnLayer::name( void ) const {
        return m_name;
    }
    
    Matrix*
    DnnLayer::outA( void ) {
        return m_out_a;             // Output Activations
    }
    
    Matrix*
    DnnLayer::weights( void ) {     // Internal Neuron connection weights    (w)
        return m_w;
    }
    
    Matrix*
    DnnLayer::gradiants( void ) {   // Internal Neuron connection gradiants (dw)
        return m_dw;
    }
    
    DNN_NUMERIC
    DnnLayer::l1DecayMultiplier( void ) const {
        return m_l1_decay_mul;
    }
    DNN_NUMERIC
    DnnLayer::l1DecayMultiplier( DNN_NUMERIC value ) {
        return m_l1_decay_mul = value;
    }
    
    DNN_NUMERIC
    DnnLayer::l2DecayMultiplier( void ) const {
        return m_l2_decay_mul;
    }
    DNN_NUMERIC
    DnnLayer::l2DecayMultiplier( DNN_NUMERIC value ) {
        return m_l2_decay_mul = value;
    }
    
    DnnLayer*
    DnnLayer::getPreviousLayer( void ) const {
        return m_prev_layer;
    }
    
    DnnLayer*
    DnnLayer::setPreviousLayer( DnnLayer *layer ) {
        if( layer != 0 ) {
            m_in_a  = layer->m_out_a;     // Remember previous activation layer
            m_in_dw = layer->m_out_dw;    // Remember previous gradiant array
        }
        return m_prev_layer = layer;
    }
    
    DnnLayer*
    DnnLayer::getNextLayer( void ) const {
        return m_next_layer;
    }
    
    DnnLayer*
    DnnLayer::setNextLayer( DnnLayer *layer ) {
        return m_next_layer = layer;
    }

    void
    DnnLayer::initialize( void ) {
        // Zero activations, gradiant and randomize weights.
        if( m_w != 0 ) {
            m_w->randomize();
        }
        if( m_dw != 0 ) {
            m_dw->zero();
        }
        if( m_out_a != 0 ) {
            m_out_a->zero();
        }
        if( m_out_dw != 0 ) {
            m_out_dw->zero();
        }
        if( m_next_layer != 0 ) {
            m_next_layer->initialize();  // Forward propagate
        }
        return;
    }

    void
    DnnLayer::randomize( void ) {
        // Randomize weights and bias.
        if( m_w != 0 ) {
            m_w->randomize();
        }
        if( m_next_layer != 0 ) {
            m_next_layer->randomize();  // Forward propagate
        }
        return;
    }

    bool
    DnnLayer::forward( void ) {
        // Forward propagate while training
        if( m_next_layer != 0 ) {
            return m_next_layer->forward();
        }
        return true;
    }

    bool
    DnnLayer::backprop( void ) {  // Back propagate while training
        if( m_prev_layer != 0 ) {
            return m_prev_layer->backprop();
        }
        return true;
    }
    
    DNN_NUMERIC
    DnnLayer::backprop( const Matrix &expectation ) {
        if( m_prev_layer != 0 ) {
            return m_prev_layer->backprop();
        }
        return 0.0;
    }

    DNN_NUMERIC
    DnnLayer::backprop( const DNN_INTEGER expectation ) {
        if( m_prev_layer != 0 ) {
            return m_prev_layer->backprop();
        }
        return 0.0;
    }
    
    bool
    DnnLayer::predict( const Matrix &data ) {
        // Forward progagate when predicting
        if( m_next_layer != 0 ) {
            return m_next_layer->predict( data );
        }
        return true;
    }

    
}   // namespace tfs
