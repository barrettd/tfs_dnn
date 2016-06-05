// --------------------------------------------------------------------
//  DnnLayerConvolution.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerConvolution.h"
#include "Error.h"

namespace tfs {
  
    static const char *NAME = "conv";

    const char*
    DnnLayerConvolution::className( void ) {
        return NAME;
    }

    DnnLayerConvolution::DnnLayerConvolution( DnnLayer *previousLayer,
                                             unsigned long side,
                                             unsigned long filterCount,
                                             unsigned long stride,
                                             unsigned long pad,
                                             const bool trainable ):
    DnnLayer( NAME, previousLayer ),
    m_side(    side ),
    m_stride(  stride ),
    m_pad(     pad ),
    m_filter_count( filterCount ) {
        m_l1_decay_mul = 0.0;
        m_l2_decay_mul = 1.0;
        if( previousLayer != 0 ) {  // previousLayer should not be null.
            setup( trainable );
        } else {
            log_error( "previousLayer is null" );
        }
    }
    
    DnnLayerConvolution::~DnnLayerConvolution( void ) {
        // Destructor
    }
    
    void
    DnnLayerConvolution::setup( const bool trainable ) {
        // -----------------------------------------------------------------------------------
        // Allocate filters and output
        // -----------------------------------------------------------------------------------
        if( m_side < 1 || m_filter_count < 1 || m_stride < 1 ) {
            log_error( "Bad params" );
            return;
        }
        if( trainable && m_in_dw != 0 ) {               // Input layers can have null m_out_dw
            if( m_in_dw == 0 ) {
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
        
        const unsigned long out_x = floorl((in_x + m_pad * 2 - m_side) / m_stride + 1 );
        const unsigned long out_y = floorl((in_y + m_pad * 2 - m_side) / m_stride + 1 );
        const unsigned long out_z = m_filter_count;
        
        m_w = new Matrix( m_filter_count, m_side, m_side, in_z + 1 );       // N Fiters of side x side x (depth +1) for the bias
        if( trainable ) {
            m_dw = new Matrix( *m_w );                      // Gradiant N x side x side + (depth+1)
        }
        m_out_a = new Matrix( out_x, out_y, out_z );        // Activations (output)
        if( trainable ) {
            m_out_dw = new Matrix( *m_out_a );              // Activation gradiant
        }
        return;
    }
    
    bool
    DnnLayerConvolution::runForward(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_out_a == 0 ) {
            return log_error( "Not configured" );
        }
        const unsigned long in_x   = m_in_a->width();        // var V_sx = V.sx |0;
        const unsigned long in_y   = m_in_a->height();       // var V_sy = V.sy |0;
        const unsigned long in_z   = m_in_a->depth();
        const unsigned long side   = m_side;
        const unsigned long stride = m_stride;              // var xy_stride = this.stride |0;
        
        const unsigned long out_x = m_out_a->width();
        const unsigned long out_y = m_out_a->height();
        const unsigned long out_z = m_out_a->depth();       // Filter count

        const unsigned long last = side-1;
        for( unsigned long d = 0; d < out_z; d++ ) {        // Filter count
            long x = -m_pad;
            for( unsigned long ax = 0; ax < out_x; x += stride, ax++ ) {
                long y = -m_pad;
                for( unsigned long ay = 0; ay < out_y; y += stride, ay++ ) {
                    // convolve centered at this point
                    DNN_NUMERIC a = 0.0;
                    for( unsigned long fx = 0; fx < side; fx++ ) {
                        for( unsigned long  fy = 0; fy < side; fy++ ) {
                            for( unsigned long fd = 0;fd < in_z; fd++ ) {
                                long oy = y+fy; // coordinates in the original input array coordinates
                                long ox = x+fx;
                                if( oy >= 0 && oy < in_y && ox >= 0 && ox < in_x ) {
                                    a += m_w->get( d, fx, fy, fd ) * m_in_a->get( ox, oy, fd );
                                }
                            }
                        }
                    }
                    a += m_w->get( d, last, last, in_z );   // bias
                    m_out_a->set( ax, ay, d, a );       // set output
                }
            }
        }
        return true;
    }
    
    bool
    DnnLayerConvolution::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_w == 0 || m_dw == 0 ) {
            return log_error( "Not configured" );
        }
        const unsigned long in_x   = m_in_a->width();        // var V_sx = V.sx |0;
        const unsigned long in_y   = m_in_a->height();       // var V_sy = V.sy |0;
        const unsigned long in_z   = m_in_a->depth();
        const unsigned long side   = m_side;
        const unsigned long stride = m_stride;              // var xy_stride = this.stride |0;
        
        const unsigned long out_x = m_out_dw->width();
        const unsigned long out_y = m_out_dw->height();
        const unsigned long out_z = m_out_dw->depth();

        if( m_in_dw != 0 ) {        // Input layer often does not have dw.
            m_in_dw->zero();        // Zero input gradiant, we add to it below.
        }
        const unsigned long last = side-1;
        for( unsigned long d = 0; d < out_z; d++ ) {
            long x = -m_pad;
            for( unsigned long ax = 0; ax < out_x; x += stride, ax++ ) {
                long y = -m_pad;
                for( unsigned long ay = 0; ay < out_y; y += stride, ay++ ) {
                    // convolve and add up the gradients.
                    const DNN_NUMERIC chain_grad = m_out_dw->get( ax, ay, d ); // gradient from chain rule
                    for( unsigned long fx = 0; fx < side; fx++ ) {
                        for( unsigned long  fy = 0; fy < side; fy++ ) {
                            for( unsigned long fd = 0;fd < in_z; fd++ ) {
                                long oy = y+fy; // coordinates in the original input array coordinates
                                long ox = x+fx;
                                if( oy >= 0 && oy < in_y && ox >= 0 && ox < in_x ) {
                                    // forward prop calculated: a += f.get(fx, fy, fd) * V.get(ox, oy, fd);
                                    // f.add_grad(fx, fy, fd, V.get(ox, oy, fd) * chain_grad);
                                    // V.add_grad(ox, oy, fd, f.get(fx, fy, fd) * chain_grad);
                                    const DNN_NUMERIC in_delta = m_in_a->get( d, ox, oy, fd ) * chain_grad;
                                    m_dw->plusEquals( d, fx, fy, fd, in_delta );
                                    if( m_in_dw != 0 ) {
                                        const DNN_NUMERIC dw_delta = m_w->get( d, fx, fy, fd ) * chain_grad;
                                        m_in_dw->plusEquals( d, ox, oy, fd, dw_delta );
                                    }
                                }
                            }
                        }
                    }
                    m_dw->plusEquals( d, last, last, in_z, chain_grad );
                }
            }
        }
        return true;
    }

    
    
}   // namespace tfs
