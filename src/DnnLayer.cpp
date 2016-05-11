// --------------------------------------------------------------------
//  DnnLayer.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include <cmath>        // sqrt()
#include <cstdlib>      // RAND_MAX
#include "DnnLayer.h"

namespace tfs {
    
    DnnLayer::DnnLayer( const char *name ):
    m_name( name ),
    m_w( 0 ), m_dw( 0 ),
    m_x( 0 ), m_y(  0 ), m_z( 0 ), m_size( 0 ),
    m_prev_layer( 0 ), m_next_layer( 0 ) {
        // Constructor
    }
    
    DnnLayer::DnnLayer( const char *name, DnnLayer *previousLayer ) :
    m_name( name ),
    m_w( 0 ), m_dw( 0 ),
    m_x( 0 ), m_y(  0 ), m_z( 0 ), m_size( 0 ),
    m_prev_layer( previousLayer ), m_next_layer( 0 ) {
        // Constructor
        if( previousLayer != 0 ) {
            previousLayer->setNextLayer( this );
        }
    }
    
    DnnLayer::~DnnLayer( void ) {
        teardown();
    }
    
    void
    DnnLayer::setup( unsigned long xx, unsigned long yy, unsigned long zz ) {
        teardown();
        m_x = xx;
        m_y = yy;
        m_z = zz;
        m_size = m_x * m_y * m_z;
        m_w  = new DNN_NUMERIC[m_size];
        m_dw = new DNN_NUMERIC[m_size];
        return;
    }
    
    void
    DnnLayer::teardown( void ) {
        delete[] m_w;
        delete[] m_dw;
        m_w    = 0;
        m_dw   = 0;
        m_size = 0;
        m_prev_layer = 0;
        m_next_layer = 0;
        return;
    }

    const char*
    DnnLayer::name( void ) const {
        return m_name;
    }
    
    unsigned long
    DnnLayer::width( void ) const {
        return m_x;
    }
    
    unsigned long
    DnnLayer::height( void ) const {
        return m_y;
    }
    
    unsigned long
    DnnLayer::depth( void ) const {
        return m_z;
    }
    
    unsigned long
    DnnLayer::size(   void ) const {
        return m_size;          // = x * w * h;  // Count of DNN_NUMERIC elements.
    }
    
    DNN_NUMERIC*
    DnnLayer::w( void ) {
        return m_w;             // weights
    }
    
    DNN_NUMERIC*
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
        // Weight normalization is done to equalize the output variance of every neuron,
        // otherwise neurons with a lot of incoming connections will have outputs with a larger variance
        if( m_w != 0 && m_size > 0 ) {
            const DNN_NUMERIC scale = sqrt( 1.0 / m_size );
            DNN_NUMERIC *ptr = m_w;
            DNN_NUMERIC *end = m_w + m_size;
            while( ptr < end ) {
                *ptr++ = ((double) rand() / (RAND_MAX)) * scale;
            }
        }
        if( m_next_layer != 0 ) {
            m_next_layer->randomize();
        }
        return;
    }

    bool
    DnnLayer::forward( const DNN_NUMERIC *data, const unsigned long length ) {
        // Forward propagate while training
        if( m_next_layer != 0 ) {
            return m_next_layer->forward( data, length );
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
    DnnLayer::predict( const DNN_NUMERIC *data, const unsigned long length ) {
        // Forward progagate when predicting
        if( m_next_layer != 0 ) {
            return m_next_layer->predict( data, length );
        }
        return true;
    }


    
}   // namespace tfs
