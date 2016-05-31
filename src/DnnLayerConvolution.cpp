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
                                             unsigned long filters,
                                             unsigned long stride,
                                             unsigned long pad,
                                             const bool trainable ):
    DnnLayer( NAME, previousLayer ),
    m_side(    side ),
    m_stride(  stride ),
    m_pad(     pad ),
    m_filter_count( filters ),
    m_filter(  0 ) {              // Constructor
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
        if( m_filter != 0 ) {
            for( unsigned long ii = 0; ii < m_filter_count; ii++ ) {
                delete m_filter[ii];
            }
            delete[] m_filter;
            m_filter = 0;
        }
    }
    
    void
    DnnLayerConvolution::setup( const bool trainable ) {
        // -----------------------------------------------------------------------------------
        // N = number of filters
        // S = size of input data
        //  w[N,S+1] = filters weight + bias weight
        // dw[N,S+1] = gradiant + d/dw bias
        // out_a[N]  = activations
        // out_dw[N] = gradiant
        // -----------------------------------------------------------------------------------
        if( m_side < 1 || m_filter_count < 1 || m_stride < 1 ) {
            log_error( "Bad params" );
            return;
        }
        if( m_in_a == 0 ) {
            log_error( "Input activation matrix is null" );
            return;
        }
        const unsigned long in_x = m_in_a->width();
        const unsigned long in_y = m_in_a->height();
        const unsigned long in_z = m_in_a->depth();
        
        const unsigned long out_x = floorl((in_x + m_pad * 2 - m_side) / m_stride + 1 );
        const unsigned long out_y = floorl((in_y + m_pad * 2 - m_side) / m_stride + 1 );
        const unsigned long out_z = m_filter_count;
        
        const unsigned long N = m_filter_count;
        const unsigned long S = m_side * m_side * in_z;     // 1d input, S elements.
        
        m_w = new Matrix( N, S+1, 1 );                      // weights N x (S+1)
        if( trainable ) {
            m_dw = new Matrix( N, S+1, 1 );                 // gradiant N x (S+1)
        }
        m_out_a = new Matrix( out_x, out_y, out_z );        // Activations (output)
        if( trainable ) {
            m_out_dw = new Matrix( out_x, out_y, out_z );   // Activation gradiant
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
        const unsigned long in_x  = m_in_a->width();        // var V_sx = V.sx |0;
        const unsigned long in_y  = m_in_a->height();       // var V_sy = V.sy |0;
        const unsigned long stride = m_stride;              // var xy_stride = this.stride |0;
        
        const unsigned long out_x = m_out_a->width();
        const unsigned long out_y = m_out_a->height();
        const unsigned long out_z = m_out_a->depth();

        const DNN_NUMERIC *input  = m_in_a->dataReadOnly();
        const DNN_NUMERIC *weight = m_w->dataReadOnly();

        for( unsigned long z = 0; z < out_z; z++ ) {
            long x = -m_pad;
            long y = -m_pad;
            for( unsigned long ay = 0; ay < out_y; y += m_stride, ay++ ) {
                x = -m_pad;
                for( unsigned long ax = 0; ax < out_x; x += m_stride, ax++ ) {
                    // Convolve centered at [ax, ay]
                    DNN_NUMERIC a = 0.0;
                    long winx = -1;
                    long winy = -1;
                    for( unsigned long fy = 0; fy < m_side; fy++ ) {
                        long oy = y + fy;
                        for( unsigned long fx = 0; fx < m_side; fx++ ) {
                            long ox = x + fx;
                            if( oy >= 0 && oy < in_y && ox >= 0 && ox < in_x ) {
                                for( unsigned long fd = 0; fd < m_filter_count; fd++ ) {
                                    //a += f.get(fx, fy, fd) * V.get(ox, oy, fd);
//                                    unsigned long filter_index = ((f.sx * fy)+fx)*f.depth+fd;
//                                    unsigned long input_index  = ((V_sx * oy)+ox)*V.depth+fd;
//                                    a += weight[filter_index] * input[input_index];
                                }
                            }
                        }
                    }
//                    m_switch[n++] = winx;
//                    m_switch[n++] = winy;
                }
            }
        }
//
//        for(var d=0;d<this.out_depth;d++) {
//            var f = this.filters[d];
//            var x = -this.pad |0;
//            var y = -this.pad |0;
//            for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
//                x = -this.pad |0;
//                for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride
//                    
//                    // convolve centered at this particular location
//                    var a = 0.0;
//                    for(var fy=0;fy<f.sy;fy++) {
//                        var oy = y+fy; // coordinates in the original input array coordinates
//                        for(var fx=0;fx<f.sx;fx++) {
//                            var ox = x+fx;
//                            if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
//                                for(var fd=0;fd<f.depth;fd++) {
//                                    // avoid function call overhead (x2) for efficiency, compromise modularity
//                                    a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V_sx * oy)+ox)*V.depth+fd];
//                                }
//                            }
//                        }
//                    }
//                    a += this.biases.w[d];
//                    A.set(ax, ay, d, a);
//                }
//            }
//        }
        
        return true;
    }
    
    bool
    DnnLayerConvolution::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // -----------------------------------------------------------------------------------
        // TODO:
//    backward: function() {
//        
//        var V = this.in_act;
//        V.dw = global.zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it
//        
//        var V_sx = V.sx |0;
//        var V_sy = V.sy |0;
//        var xy_stride = this.stride |0;
//        
//        for(var d=0;d<this.out_depth;d++) {
//            var f = this.filters[d];
//            var x = -this.pad |0;
//            var y = -this.pad |0;
//            for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
//                x = -this.pad |0;
//                for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride
//                    
//                    // convolve centered at this particular location
//                    var chain_grad = this.out_act.get_grad(ax,ay,d); // gradient from above, from chain rule
//                    for(var fy=0;fy<f.sy;fy++) {
//                        var oy = y+fy; // coordinates in the original input array coordinates
//                        for(var fx=0;fx<f.sx;fx++) {
//                            var ox = x+fx;
//                            if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
//                                for(var fd=0;fd<f.depth;fd++) {
//                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
//                                    var ix1 = ((V_sx * oy)+ox)*V.depth+fd;
//                                    var ix2 = ((f.sx * fy)+fx)*f.depth+fd;
//                                    f.dw[ix2] += V.w[ix1]*chain_grad;
//                                    V.dw[ix1] += f.w[ix2]*chain_grad;
//                                }
//                            }
//                        }
//                    }
//                    this.biases.dw[d] += chain_grad;
//                }
//            }
//        }
        return true;
    }

    
    
}   // namespace tfs