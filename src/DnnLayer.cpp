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
    m_pa( 0 ), m_w( 0 ), m_dw( 0 ), m_a( 0 ),
    m_prev_layer( 0 ), m_next_layer( 0 ) {
        // Constructor
    }
    
    DnnLayer::DnnLayer( const char *name, DnnLayer *previousLayer ) :
    m_name( name ),
    m_pa( 0 ), m_w( 0 ), m_dw( 0 ), m_a( 0 ),
    m_prev_layer( previousLayer ), m_next_layer( 0 ) {
        // Constructor
        if( previousLayer != 0 ) {
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
    DnnLayer::setup( const unsigned long  inX, const unsigned long  inY, const unsigned long  inZ,
                     const unsigned long outX, const unsigned long outY, const unsigned long outZ,
                     const bool trainable ) {
        teardown();
        m_w = new Matrix( inX, inY, inZ );
        if( trainable ) {
            m_dw = new Matrix( inX, inY, inZ );
        }
        m_a = new Matrix( outX, outY, outZ );
        return;
    }
    
    void
    DnnLayer::setup( const Matrix *activations, const bool trainable ) {
        if( activations == 0 ) {
            log_error( "activations are null" );
            return;
        }
        const unsigned long  xx = activations->width();
        const unsigned long  yy = activations->height();
        const unsigned long  zz = activations->depth();
        
        setup( xx, yy, zz, xx, yy, zz, trainable );
        return;
    }

    void
    DnnLayer::teardown( void ) {
        delete m_w;
        delete m_dw;
        delete m_a;
        m_pa   = 0;
        m_w    = 0;
        m_dw   = 0;
        m_a    = 0;
        return;
    }

    const char*
    DnnLayer::name( void ) const {
        return m_name;
    }
    
    Matrix*
    DnnLayer::w( void ) {
        return m_w;             // Weights
    }
    
    Matrix*
    DnnLayer::dw( void ) {
        return m_dw;            // Weight derivatives
    }
    
    Matrix*
    DnnLayer::a( void ) {
        return m_a;             // Activations
    }
    
    unsigned long
    DnnLayer::aX( void ) const {
        if( m_a == 0 ) {
            return 0;
        }
        return m_a->width();
    }
    
    unsigned long
    DnnLayer::aY( void ) const {
        if( m_a == 0 ) {
            return 0;
        }
        return m_a->height();
    }

    unsigned long
    DnnLayer::aZ( void ) const {
        if( m_a == 0 ) {
            return 0;
        }
        return m_a->depth();
    }
    
    unsigned long
    DnnLayer::aSize( void ) const {
        if( m_a == 0 ) {
            return 0;
        }
        return m_a->size();
    }

    DnnLayer*
    DnnLayer::getPreviousLayer( void ) const {
        return m_prev_layer;
    }
    
    DnnLayer*
    DnnLayer::setPreviousLayer( DnnLayer *layer ) {
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
        if( m_a != 0 ) {
            m_a->zero();
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
    DnnLayer::forward( const Matrix &data ) {
        // Forward propagate while training
        if( m_next_layer != 0 ) {
            return m_next_layer->forward( data );
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

    bool
    DnnLayer::predict( const Matrix &data ) {
        // Forward progagate when predicting
        if( m_next_layer != 0 ) {
            return m_next_layer->predict( data );
        }
        return true;
    }


    
}   // namespace tfs
