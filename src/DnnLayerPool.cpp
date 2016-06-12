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
    
    static const char *NAME = "pool";

    const char*
    DnnLayerPool::className( void ) {
        return NAME;
    }

    DnnLayerPool::DnnLayerPool( DnnLayer *previousLayer, unsigned long side, unsigned long stride, unsigned long pad, const bool trainable ):
    DnnLayer( NAME, previousLayer ),
    m_side(     side ),
    m_stride( stride ),
    m_pad(       pad ),
    m_switch(      0 ) {
        // Constructor
        setup( trainable );
        
    }
    
    DnnLayerPool::~DnnLayerPool( void ) {
        // Destructor
        delete[] m_switch;
        m_switch = 0;
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
        
        m_switch = new unsigned long[ out_x * out_y * out_z * 2 ];
        
        m_out_a = new Matrix( out_x, out_y, out_z );
        if( trainable ) {
            m_out_dw = new Matrix( *m_out_a );          // dw dimenstion matches a
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
        
        m_out_a->zero();        // May not need to do this.
        
        unsigned long *switches = m_switch;     // Pointer for switches.
        for( unsigned long z = 0; z < out_z; z++ ) {
            long y = - (long) m_pad;
            for( unsigned long ay = 0; ay < out_y; y += m_stride, ay++ ) {
                long x = - (long) m_pad;
                for( unsigned long ax = 0; ax < out_x; x += m_stride, ax++ ) {
                    // Convolve centered at [ax, ay]
                    DNN_NUMERIC a = -__DBL_MAX__;
                    unsigned long winx = 0;
                    unsigned long winy = 0;
                    for( unsigned long fy = 0; fy < m_side; fy++ ) {
                        long oy = y + (long) fy;
                        for( unsigned long fx = 0; fx < m_side; fx++ ) {
                            long ox = x + (long) fx;
                            if( oy >= 0 && oy < (long) in_y && ox >= 0 && ox < (long) in_x ) {
                                DNN_NUMERIC v = m_in_a->get((unsigned long) ox, (unsigned long) oy, z );
                                if( v > a ) {   // perform max pooling and store indexes to where
                                    a = v;      // the max came from. This will speed up backprop
                                    winx = (unsigned long) ox;
                                    winy = (unsigned long) oy;
                                }
                            }
                        }
                    }
                    m_out_a->set( ax, ay, z, a );
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

        m_in_dw->zero();
        const unsigned long *switches = m_switch;     // Pointer for switches.
        for( unsigned long z = 0; z < out_z; z++ ) {
            long y = - (long) m_pad;
            for( unsigned long ay = 0; ay < out_y; y += m_stride, ay++ ) {
                long x = - (long) m_pad;
                for( unsigned long ax = 0; ax < out_x; x += m_stride, ax++ ) {
                    const DNN_NUMERIC chain_grad = m_out_dw->get( ax, ay, z );
                    const unsigned long ox = *switches++;
                    const unsigned long oy = *switches++;
                    m_in_dw->set( ox, oy, z, chain_grad );
                }
            }
        }
        return true;
    }
    

}   // namespace tfs

