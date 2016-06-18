// --------------------------------------------------------------------
//  DnnLayerConvolution.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerConvolution.h"
#include "Error.h"

namespace tfs {
  
    DnnLayerConvolution::DnnLayerConvolution( DnnLayer *previousLayer,
                                             unsigned long side,
                                             unsigned long filterCount,
                                             unsigned long stride,
                                             unsigned long pad,
                                             const bool trainable ):
    DnnLayer( LAYER_CONVOLUTION, previousLayer ),
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
    
    unsigned long
    DnnLayerConvolution::side( void ) const {
        return m_side;
    }
    unsigned long
    DnnLayerConvolution::stride( void ) const {
        return m_stride;
    }
    unsigned long
    DnnLayerConvolution::pad( void ) const {
        return m_pad;
    }
    unsigned long
    DnnLayerConvolution::filterCount( void ) const {
        return m_filter_count;
    }

    void
    DnnLayerConvolution::setup( const bool trainable ) {
        // -----------------------------------------------------------------------------------
        // Allocate filters and output
        // out_z == filter_count
        // -----------------------------------------------------------------------------------
        if( m_side < 1 || m_filter_count < 1 || m_stride < 1 ) {
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
        const unsigned long out_z = m_filter_count;
        
        m_w = new Matrix( m_side, m_side, in_z, m_filter_count );   // side x side x depth x N filters
        if( trainable ) {
            m_dw = new Matrix( *m_w );                      // Gradiant side x side + depth x N filters
        }
        m_bias_w = new Matrix( out_z );
        if( trainable ) {
            m_bias_dw = new Matrix( *m_bias_w );
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
        // ok 11 June 2016
        // TODO: Speed up with pointers. (benchmark A/B compare.)
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

        for( unsigned long az = 0; az < out_z; az++ ) {     // Filter count
            long yy = - (long) m_pad;
            for( unsigned long ay = 0; ay < out_y; ay++, yy += stride ) {
                long xx = - (long) m_pad;
                for( unsigned long ax = 0; ax < out_x; ax++, xx += stride ) {
                    // convolve centered at this point
                    DNN_NUMERIC a = 0.0;
                    for( unsigned long  fy = 0; fy < side; fy++ ) {
                        long oy = yy + (long) fy;
                        for( unsigned long fx = 0; fx < side; fx++ ) {
                            long ox = xx + (long) fx;
                            if( oy >= 0 && oy < (long) in_y && ox >= 0 && ox < (long) in_x ) {
                                for( unsigned long fz = 0; fz < in_z; fz++ ) {
                                    a += m_w->get( fx, fy, fz, az ) * m_in_a->get((unsigned long) ox, (unsigned long)oy, fz );
                                }
                            }
                        }
                    }
                    a += m_bias_w->get( az );        // bias
                    m_out_a->set( ax, ay, az, a );   // set output
                }
            }
        }
        return true;
    }
    
    bool
    DnnLayerConvolution::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // This routine is the biggest time sink in the DigitRecognizer app.
        // ok 11 June 2016
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
        for( unsigned long az = 0; az < out_z; az++ ) {
            long yy = -(long) m_pad;
            for( unsigned long ay = 0; ay < out_y; ay++, yy += stride ) {
                long xx = -(long) m_pad;
                for( unsigned long ax = 0; ax < out_x; ax++, xx += stride ) {
                    // convolve and add up the gradients.
                    const DNN_NUMERIC chain_grad = m_out_dw->get( ax, ay, az );  // gradient from chain rule
                    for( unsigned long  fy = 0; fy < side; fy++ ) {
                        long oy = yy + (long) fy;
                        for( unsigned long fx = 0; fx < side; fx++ ) {
                            long ox = xx + (long) fx;
                            if( oy >= 0 && oy < (long) in_y && ox >= 0 && ox < (long) in_x ) {
                               for( unsigned long fz = 0; fz < in_z; fz++ ) {
                                   unsigned long ix1 = m_in_a->getIndex((unsigned long) ox, (unsigned long) oy, fz );
                                   unsigned long ix2 = m_w->getIndex( fx, fy, fz, az );
                                   const DNN_NUMERIC value1 = m_in_a->get( ix1 ) * chain_grad;
                                   const DNN_NUMERIC value2 = m_w->get(    ix2 ) * chain_grad;
                                   m_dw->plusEquals( ix2, value1 );
                                   if( m_in_dw != 0 ) {
                                      m_in_dw->plusEquals( ix1, value2 );
                                   }
//                                   const DNN_NUMERIC in_delta = m_in_a->get((unsigned long)ox, (unsigned long)oy, fz ) * chain_grad;
//                                    m_dw->plusEquals( fx, fy, fz, az, in_delta );
//                                    if( m_in_dw != 0 ) {
//                                        const DNN_NUMERIC dw_delta = m_w->get( fx, fy, fz, az ) * chain_grad;
//                                        m_in_dw->plusEquals((unsigned long)ox, (unsigned long)oy, fz, dw_delta );
//                                    }
                                }
                            }
                        }
                    }
                    m_bias_dw->plusEquals( az, chain_grad );
                }
            }
        }
        return true;
    }

    
    
}   // namespace tfs
