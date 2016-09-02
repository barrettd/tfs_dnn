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
            m_dw = new Matrix( *m_w );                      // Gradient side x side + depth x N filters
        }
        m_bias_w = new Matrix( out_z );
        if( trainable ) {
            m_bias_dw = new Matrix( *m_bias_w );
        }
        m_out_a = new Matrix( out_x, out_y, out_z );        // Activations (output)
        if( trainable ) {
            m_out_dw = new Matrix( *m_out_a );              // Activation gradient
        }
        return;
    }

    bool
    DnnLayerConvolution::runForward(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // ok 11 June 2016
        // Speed up with pointers. (benchmark A/B compare.) 22 June 2016
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_out_a == 0 ) {
            return log_error( "Not configured" );
        }
        const unsigned long in_x   = m_in_a->width();
        const unsigned long in_y   = m_in_a->height();
        const unsigned long in_z   = m_in_a->depth();
        const unsigned long side   = m_side;
        const unsigned long stride = m_stride;
        
        const unsigned long out_x = m_out_a->width();
        const unsigned long out_y = m_out_a->height();
        const unsigned long out_z = m_out_a->depth();       // Filter count
        
        const unsigned long ix1_delta  = m_in_a->ab();
        const unsigned long ix2_delta  = m_w->ab();
        
        const unsigned long filter_aa  = m_w->aa();
        const unsigned long filter_abc = m_w->abc();

        const DNN_NUMERIC *inA    = m_in_a->dataReadOnly();
        const DNN_NUMERIC *filter = m_w->dataReadOnly();
        const DNN_NUMERIC *bias   = m_bias_w->dataReadOnly();
              DNN_NUMERIC *outA   = m_out_a->data();

        for( unsigned long az = 0; az < out_z; az++ ) {     // Filter count
            long yy = - (long) m_pad;
            for( unsigned long ay = 0; ay < out_y; ay++, yy += stride ) {
                long xx = - (long) m_pad;
                for( unsigned long ax = 0; ax < out_x; ax++, xx += stride ) {
                    // convolve centered at this point
                    DNN_NUMERIC aa = 0.0;
                    for( unsigned long  fy = 0; fy < side; fy++ ) {
                        const long oy = yy + (long) fy;
                        for( unsigned long fx = 0; fx < side; fx++ ) {
                            const long ox = xx + (long) fx;
                            if( oy >= 0 && oy < (long) in_y && ox >= 0 && ox < (long) in_x ) {
                                unsigned long ix1 = oy * in_x + ox;                             // m_in_a( ox, oy, 0 )
                                unsigned long ix2 = az * filter_abc + fy * filter_aa + fx;      // m_w( fx, fy, 0, az );
                                for( unsigned long fz = 0; fz < in_z; fz++ ) {
                                    aa += filter[ix2] * inA[ix1];
                                    ix1 += ix1_delta;
                                    ix2 += ix2_delta;
                                }
                            }
                        }
                    }
                    aa += *bias;        // m_bias_w->get( az );        // bias
                    *outA++ = aa;       // m_out_a->set( ax, ay, az, aa );  // set output
                }
            }
            bias++;
        }
        return true;
    }
    
    bool
    DnnLayerConvolution::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // This routine is the biggest time sink in the DigitRecognizer app.
        // ok 11 June 2016
        // 61.1% -> 17.5% 22 June 2016
        // -----------------------------------------------------------------------------------
        if( m_in_a == 0 || m_w == 0 || m_dw == 0 ) {
            return log_error( "Not configured" );
        }
        const unsigned long in_x     = m_in_a->width();
        const unsigned long in_y     = m_in_a->height();
        const unsigned long in_z     = m_in_a->depth();
        const unsigned long side     = m_side;
        const unsigned long stride   = m_stride;
        const          long padStart = -(long) m_pad;
        
        const unsigned long out_x = m_out_dw->width();
        const unsigned long out_y = m_out_dw->height();
        const unsigned long out_z = m_out_dw->depth();
        
        const unsigned long ix1_delta  = m_in_a->ab();
        const unsigned long ix2_delta  = m_w->ab();
        
        const unsigned long filter_aa  = m_w->aa();
        const unsigned long filter_abc = m_w->abc();
        
        const DNN_NUMERIC *inA      = m_in_a->dataReadOnly();
        const DNN_NUMERIC *filter   = m_w->dataReadOnly();
              DNN_NUMERIC *filterDw = m_dw->data();
        const DNN_NUMERIC *outDw    = m_out_dw->dataReadOnly();
              DNN_NUMERIC *biasDw   = m_bias_dw->data();         // [out_z]
        
        if( m_in_dw != 0 ) {
            m_in_dw->zero();                                    // Zero input gradient, we add to it below.
            DNN_NUMERIC *inDw = m_in_dw->data();
            for( unsigned long az = 0; az < out_z; az++ ) {
                long yy = padStart;
                for( unsigned long ay = 0; ay < out_y; ay++, yy += stride ) {
                    long xx = padStart;
                    for( unsigned long ax = 0; ax < out_x; ax++, xx += stride ) {
                        // convolve and add up the gradients.
                        const DNN_NUMERIC chain_grad = *outDw++;        // m_out_dw( ax, ay, az ) : chain rule
                        for( unsigned long  fy = 0; fy < side; fy++ ) {
                            const long oy = yy + (long) fy;
                            for( unsigned long fx = 0; fx < side; fx++ ) {
                                const long ox = xx + (long) fx;
                                if( oy >= 0 && oy < (long) in_y && ox >= 0 && ox < (long) in_x ) {
                                    unsigned long ix1 = oy * in_x + ox;                             // m_in_a( ox, oy, 0 )
                                    unsigned long ix2 = az * filter_abc + fy * filter_aa + fx;      // m_w( fx, fy, 0, az );
                                    for( unsigned long fz = 0; fz < in_z; fz++ ) {
                                        filterDw[ix2] += inA[   ix1] * chain_grad;
                                        inDw[    ix1] += filter[ix2] * chain_grad;
                                        ix1 += ix1_delta;
                                        ix2 += ix2_delta;
                                    }
                                }
                            }
                        }
                        *biasDw += chain_grad;
                    }
                }
                biasDw++;
            }
        } else {
            for( unsigned long az = 0; az < out_z; az++ ) {
                long yy = padStart;
                for( unsigned long ay = 0; ay < out_y; ay++, yy += stride ) {
                    long xx = padStart;
                    for( unsigned long ax = 0; ax < out_x; ax++, xx += stride ) {
                        // convolve and add up the gradients.
                        const DNN_NUMERIC chain_grad = *outDw++;        // m_out_dw->get( ax, ay, az );  // gradient from chain rule
                        for( unsigned long  fy = 0; fy < side; fy++ ) {
                            const long oy = yy + (long) fy;
                            for( unsigned long fx = 0; fx < side; fx++ ) {
                                const long ox = xx + (long) fx;
                                if( oy >= 0 && oy < (long) in_y && ox >= 0 && ox < (long) in_x ) {
                                    unsigned long ix1 = oy * in_x + ox;                             // m_in_a( ox, oy, 0 )
                                    unsigned long ix2 = az * filter_abc + fy * filter_aa + fx;      // m_w( fx, fy, 0, az );
                                    for( unsigned long fz = 0; fz < in_z; fz++ ) {
                                        filterDw[ix2] += inA[ix1] * chain_grad;
                                        ix1 += ix1_delta;
                                        ix2 += ix2_delta;
                                    }
                                }
                            }
                        }
                        *biasDw += chain_grad;
                    }
                }
                biasDw++;
            }
        }
        return true;
    }

    
    
}   // namespace tfs
