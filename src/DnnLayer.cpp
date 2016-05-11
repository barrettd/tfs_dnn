// --------------------------------------------------------------------
//  DnnLayer.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayer.h"

namespace tfs {
    
    DnnLayer::DnnLayer( const char *name ):
    m_name( name ),
    m_w( 0 ), m_dw( 0 ),
    m_prev_layer( 0 ), m_next_layer( 0 ) {
        // Constructor
    }
    
    DnnLayer::DnnLayer( const char *name, DnnLayer *previousLayer ) :
    m_name( name ),
    m_w( 0 ), m_dw( 0 ),
    m_prev_layer( previousLayer ), m_next_layer( 0 ) {
        // Constructor
        if( previousLayer != 0 ) {
            previousLayer->setNextLayer( this );
        }
    }
    
    DnnLayer::~DnnLayer( void ) {
        teardown();
        m_prev_layer = 0;
        m_next_layer = 0;
    }
    
    void
    DnnLayer::setup( unsigned long xx, unsigned long yy, unsigned long zz, bool training ) {
        teardown();
        m_w = new Matrix( xx, yy, zz );
        if( training ) {
            m_dw = new Matrix( xx, yy, zz );
        }
        return;
    }
    
    void
    DnnLayer::teardown( void ) {
        delete m_w;
        delete m_dw;
        m_w    = 0;
        m_dw   = 0;
        return;
    }

    const char*
    DnnLayer::name( void ) const {
        return m_name;
    }
    
    Matrix*
    DnnLayer::w( void ) {
        return m_w;             // weights
    }
    
    Matrix*
    DnnLayer::dw( void ) {
        return m_dw;            // weight derivatives
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
