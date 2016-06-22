// --------------------------------------------------------------------
//  DnnLayerPool.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include <cfloat>
#include "DnnLayerPool.h"
#include "Error.h"

namespace tfs {
    
    DnnLayerPool::DnnLayerPool( DnnLayer *previousLayer, unsigned long side, unsigned long stride, unsigned long pad, const bool trainable ):
    DnnLayer( LAYER_POOL, previousLayer ),
    m_side(      side ),
    m_stride(  stride ),
    m_pad(        pad ),
    m_switch_count( 0 ),
    m_switch(       0 ) {
        // Constructor
        setup( trainable );
        
    }
    
    DnnLayerPool::~DnnLayerPool( void ) {
        // Destructor
        delete[] m_switch;
        m_switch = 0;
    }
    
    unsigned long
    DnnLayerPool::side( void ) const {
        return m_side;
    }
    unsigned long
    DnnLayerPool::stride( void ) const {
        return m_stride;
    }
    unsigned long
    DnnLayerPool::pad( void ) const {
        return m_pad;
    }
    unsigned long
    DnnLayerPool::switchCount( void ) const {
        return m_switch_count;
    }
    
    const unsigned long*
    DnnLayerPool::switchsReadOnly( void ) const {
        return m_switch;
    }
    unsigned long*
    DnnLayerPool::switchs( void ) {
        return m_switch;
    }
    
    void
    DnnLayerPool::setup( const bool trainable ) {
        // -----------------------------------------------------------------------------------
        // Convolutional pooling, using a square sample area.
        // -----------------------------------------------------------------------------------
        if( m_side < 1 || m_stride < 1 ) {
            log_error( "Bad params" );
            return;
        }
        if( matrixBad( m_in_a )) {
            log_error( "Input activation matrix is bad" );
            return;
        }
        if( trainable && m_in_dw != 0 ) {               // Input layers can have null m_out_dw
            if( matrixBad( m_in_dw )) {
                log_error( "Input dw matrix is bad" );
                return;
            }
            if( m_in_a->count() != m_in_dw->count()) {  // By default, we expect the dimensions to be the same.
                log_error( "in_a != in_dw size" );
                return;
            }
        }
        const unsigned long in_x = m_in_a->width();
        const unsigned long in_y = m_in_a->height();
        const unsigned long in_z = m_in_a->depth();
        
        const unsigned long out_x = (unsigned long) floor((in_x + m_pad * 2.0 - m_side) / m_stride + 1.0 );
        const unsigned long out_y = (unsigned long) floor((in_y + m_pad * 2.0 - m_side) / m_stride + 1.0 );
        const unsigned long out_z = in_z;
        
        m_switch_count = out_x * out_y * out_z * 2;
        m_switch = new unsigned long[ m_switch_count ];
        
        m_out_a = new Matrix( out_x, out_y, out_z );
        if( trainable ) {
            m_out_dw = new Matrix( *m_out_a );          // dw dimension matches a
        }
        return;
    }

    bool
    DnnLayerPool::runForward(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // Giving indexes a try, compared to pointers for speed.
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_out_a == 0 ) {
            return log_error( "Not configured" );
        }
        const unsigned long in_x  = m_in_a->width();
        const unsigned long in_y  = m_in_a->height();
        const unsigned long out_x = m_out_a->width();
        const unsigned long out_y = m_out_a->height();
        const unsigned long out_z = m_out_a->depth();
        
        DNN_NUMERIC *outA = m_out_a->data();
        
        m_out_a->zero();        // May not need to do this.
        
        unsigned long *switches = m_switch;     // Pointer for switches.
        for( unsigned long az = 0; az < out_z; az++ ) {
            long yy = - (long) m_pad;
            for( unsigned long ay = 0; ay < out_y; ay++, yy += m_stride ) {
                long xx = - (long) m_pad;
                for( unsigned long ax = 0; ax < out_x; ax++, xx += m_stride ) {
                    // Convolve centered at [ax, ay]
                    DNN_NUMERIC aa = -__DBL_MAX__;
                    unsigned long winx = 0;
                    unsigned long winy = 0;
                    for( unsigned long fy = 0; fy < m_side; fy++ ) {
                        const long oy = yy + (long) fy;
                        for( unsigned long fx = 0; fx < m_side; fx++ ) {
                            const long ox = xx + (long) fx;
                            if( oy >= 0 && oy < (long) in_y && ox >= 0 && ox < (long) in_x ) {
                                const DNN_NUMERIC vv = m_in_a->get((unsigned long) ox, (unsigned long) oy, az );
                                if( vv > aa ) {   // perform max pooling and store indexes to where
                                    aa = vv;      // the max came from. This will speed up backprop
                                    winx = (unsigned long) ox;
                                    winy = (unsigned long) oy;
                                }
                            }
                        }
                    }
                    *outA++ = aa;           // m_out_a->set( ax, ay, az, aa );
                    *switches++ = winx;
                    *switches++ = winy;
                }
            }
        }
        return true;
    }
    
    bool
    DnnLayerPool::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // Giving indexes a try, compared to pointers for speed.
        // -----------------------------------------------------------------------------------
        if( m_in_dw == 0 || m_out_dw == 0 ) {
            return log_error( "Not configured" );
        }
        const unsigned long out_x = m_out_dw->width();
        const unsigned long out_y = m_out_dw->height();
        const unsigned long out_z = m_out_dw->depth();
        
        const DNN_NUMERIC *outDw = m_out_dw->dataReadOnly();

        m_in_dw->zero();
        const unsigned long *switches = m_switch;     // Pointer for switches.
        for( unsigned long az = 0; az < out_z; az++ ) {
            for( unsigned long ay = 0; ay < out_y; ay++ ) {
                for( unsigned long ax = 0; ax < out_x; ax++ ) {
                    const DNN_NUMERIC chain_grad = *outDw++;            // m_out_dw->get( ax, ay, az );
                    const unsigned long winx = *switches++;
                    const unsigned long winy = *switches++;
                    m_in_dw->plusEquals( winx, winy, az, chain_grad );
                }
            }
        }
        return true;
    }
    

}   // namespace tfs

