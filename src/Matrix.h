// --------------------------------------------------------------------
//  Matrix.h - 3 or 4 D matrix <template> for DNN_NUMERIC and DNN_INTEGER
//  Possibly use Eigen matricies: https://eigen.tuxfamily.org/
//  Created by Barrett Davis on 5/10/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
// TODO: add flip horizontal and flip vertical, subsample
#ifndef Matrix_h
#define Matrix_h

#include <cmath>        // sqrt()
#include <cstring>      // memset()
#include "Constants.h"
#include "Error.h"
#include "Utility.h"    // random()

namespace tfs {         // Tree Frog Software

    template <typename T> class TMatrix  {      // Row major order (aa changes fastest, dd changes slowest.)
    protected:
        unsigned long m_a;      // x Width  - columns (contigious)
        unsigned long m_b;      // y Height - rows
        unsigned long m_c;      // z Depth
        unsigned long m_d;      // 4th dimension
        unsigned long m_ab;     // = aa * bb
        unsigned long m_abc;    // = aa * bb * cc
        unsigned long m_count;  // = aa * bb * cc * dd;  Count of T elements.
        unsigned long m_length; // = count * sizeof( T )
        T            *m_data;   // Pointer to data[] array:     = &data[0];
        T            *m_end;    // Pointer to end of the array: = &data[m_count];
        
        void allocate( void ) {
            if( m_count > 0 ) {
                m_length = m_count * sizeof( T );   // Length in bytes
                m_data   = new T[m_count];          // = &data[0];          Possibly random values.
                m_end    = m_data + m_count;        // = &m_data[m_count];
            }
        }

    public:
        TMatrix( const TMatrix &other, bool copyOther = false ) :       // Constructor
        m_a(  other.m_a ),  m_b(   other.m_b ),   m_c(     other.m_c ), m_d( other.m_d ),
        m_ab( other.m_ab ), m_abc( other.m_abc ), m_count( other.m_count ),
        m_length( 0 ), m_data( 0 ), m_end( 0 ) {
            allocate();
            if( copyOther ) {
                copy( other );
            }
        }

        TMatrix( const unsigned long aa, const unsigned long bb = 1, const unsigned long cc = 1, const unsigned long dd = 1 ):  // Constructor
        m_a( aa ), m_b( bb ), m_c( cc ), m_d( dd ),
        m_ab( aa * bb ), m_abc( aa * bb * cc ), m_count( aa * bb * cc * dd ),
        m_length( 0 ), m_data( 0 ), m_end( 0 ) {
            allocate();
        }
        
        ~TMatrix( void ) {      // Destructor
            delete[] m_data;
            m_data   = 0;
            m_end    = 0;
            m_count  = 0;
            m_length = 0;
        }
        
        inline bool ok( void ) const { return m_data != 0 && m_data <= m_end; }
        
        inline unsigned long aa(     void ) const { return m_a; }       // x
        inline unsigned long bb(     void ) const { return m_b; }       // y
        inline unsigned long cc(     void ) const { return m_c; }       // z
        inline unsigned long dd(     void ) const { return m_d; }
        
        inline unsigned long width(  void ) const { return m_a; }       // aa, x
        inline unsigned long height( void ) const { return m_b; }       // bb, y
        inline unsigned long depth(  void ) const { return m_c; }       // cc, z
 
        inline unsigned long ab(     void ) const { return m_ab;  }     // aa * bb
        inline unsigned long abc(    void ) const { return m_abc; }     // aa * bb * cc
        inline unsigned long count(  void ) const { return m_count;  }  // Count of elements = aa * bb * cc * dd;
        inline unsigned long length( void ) const { return m_length; }  // Length in bytes   = count * sizeof( T );
        
        inline bool isEmpty( void ) const { return m_data == 0 || m_data >= m_end || m_count < 1 || m_length < 1; }
        
        inline       T* data( void ) {               return m_data; }   // Start of data array
        inline const T* end(  void ) const {         return m_end;  }   // End of data array
        inline const T* dataReadOnly( void ) const { return m_data; }   // Start of data array (const)
        
        inline void randomize( void ) {                // Fill matrix with small positive values
            // Weight normalization is done to equalize the output variance of every neuron,
            // otherwise neurons with a lot of incoming connections will have outputs with a larger variance
            if( isEmpty()) {
                log_error( "empty matrix" );
                return;
            }
            const double  scale = sqrt( 1.0 / m_count );
                  T *       ptr = m_data;
            const T * const end = m_end;
            while( ptr < end ) {
                *ptr++ = (T) randn( scale );
            }
        }

        inline void zero( void ) {                  // Fill matrix with zeros
            if( m_data != 0 && m_length > 0 ) {
                memset( m_data, 0, m_length );      // Yields IEEE 0 for both integer and real valued variables.
            }
        }
        
        inline void copy( const TMatrix &matrix ) {        // Copy the contents another matrix.
            if( m_data != 0 && matrix.m_data != 0 && m_length > 0  && m_length == matrix.m_length ) {
                memcpy( m_data, matrix.m_data, m_length );
            }
        }
        
        inline bool equal( const TMatrix &matrix ) const { // Return true if matricies same dimension and contents.
            if( m_data == 0 || matrix.m_data == 0 || m_length < 1 || m_length != matrix.m_length ||
               m_a != matrix.m_a || m_b != matrix.m_b || m_c != matrix.m_c || m_d != matrix.m_d ) {
                return false;
            }
            return memcmp( m_data, matrix.m_data, m_length ) == 0;
        }
        
        inline unsigned long getIndex( const unsigned long aa,
                                       const unsigned long bb,
                                       const unsigned long cc,
                                       const unsigned long dd ) const {
            const unsigned long index = dd * m_abc + cc * m_ab + bb * m_a + aa;
            if( index >= m_count ) {
                log_error( "Index out of range: %lu/%lu, a = %lu/%lu, b = %lu/%lu, c = %lu/%lu, d = %lu/%lu",
                          index, m_count, aa, m_a, bb, m_b, cc, m_c, dd, m_d );
            }
            return index;
        }

        inline unsigned long getIndex( const unsigned long aa,
                                       const unsigned long bb,
                                       const unsigned long cc ) const {     // Assumes dd == 0
            const unsigned long index = cc * m_ab + bb * m_a + aa;
            if( index >= m_count ) {
                log_error( "Index out of range: %lu/%lu, a = %lu/%lu, b = %lu/%lu, c = %lu/%lu",
                          index, m_count, aa, m_a, bb, m_b, cc, m_c );
            }
            return index;
        }

        inline unsigned long getIndex( const unsigned long aa,
                                       const unsigned long bb ) const {     // Assumes dd == 0, cc == 0
            const unsigned long index = bb * m_a + aa;
            if( index >= m_count ) {
                log_error( "Index out of range: %lu/%lu, a = %lu/%lu, b = %lu/%lu",
                          index, m_count, aa, m_a, bb, m_b );
            }
            return index;
        }
        
        inline unsigned long getIndex( const unsigned long aa ) const {     // Assumes dd == 0, cc == 0, bb == 0
            if( aa >= m_count ) {
                log_error( "Index out of range: %lu/%lu", aa, m_count );
            }
            return aa;
        }
        
        inline T* getPtr( const unsigned long aa, const unsigned long bb, const unsigned long cc, const unsigned long dd ) {
            const unsigned long index = getIndex( aa, bb, cc, dd );
            return &m_data[index];
        }
        inline T* getPtr( const unsigned long aa, const unsigned long bb, const unsigned long cc ) {
            const unsigned long index = getIndex( aa, bb, cc );
            return &m_data[index];
        }
        inline T* getPtr( const unsigned long aa, const unsigned long bb ) {
            const unsigned long index = getIndex( aa, bb );
            return &m_data[index];
        }
        inline T* getPtr( const unsigned long aa ) {
            const unsigned long index = getIndex( aa );
            return &m_data[index];
        }
        inline const T* getPtr( const unsigned long aa, const unsigned long bb, const unsigned long cc, const unsigned long dd ) const {
            const unsigned long index = getIndex( aa, bb, cc, dd );
            return &m_data[index];
        }
        inline const T* getPtr( const unsigned long aa, const unsigned long bb, const unsigned long cc ) const {
            const unsigned long index = getIndex( aa, bb, cc );
            return &m_data[index];
        }
        inline const T* getPtr( const unsigned long aa, const unsigned long bb ) const {
            const unsigned long index = getIndex( aa, bb );
            return &m_data[index];
        }
        inline const T* getPtr( const unsigned long aa ) const {
            const unsigned long index = getIndex( aa );
            return &m_data[index];
        }

        inline T get( const unsigned long aa, const unsigned long bb, const unsigned long cc, const unsigned long dd ) const {
            const unsigned long index = getIndex( aa, bb, cc, dd );
            return m_data[index];
        }
        
        inline T get( const unsigned long aa, const unsigned long bb, const unsigned long cc ) const {
            const unsigned long index = getIndex( aa, bb, cc );
            return m_data[index];
        }
        
        inline T get( const unsigned long aa, const unsigned long bb ) const {
            const unsigned long index = getIndex( aa, bb );
            return m_data[index];
        }

        inline T get( const unsigned long aa ) const {
            const unsigned long index = getIndex( aa );
            return m_data[index];
        }

        inline T set( const unsigned long aa, const unsigned long bb, const unsigned long cc, const unsigned long dd, const T value ) {
            const unsigned long index = getIndex( aa, bb, cc, dd );
            return m_data[index] = value;
        }

        inline T set( const unsigned long x, const unsigned long y, const unsigned long z, const T value ) {
            const unsigned long index = getIndex( x, y, z );
            return m_data[index] = value;
        }

        inline T set( const unsigned long x, const unsigned long y, const T value ) {
            const unsigned long index = getIndex( x, y );
            return m_data[index] = value;
        }

        inline T set( const unsigned long x, const T value ) {
            const unsigned long index = getIndex( x );
            return m_data[index] = value;
        }
        
        inline T set( const T value ) {         // Set all elements to 'value'
                  T *      data = m_data;
            const T * const end = m_end;
            while( data < end ) {
                *data++ = value;
            }
            return value;
        }

        inline T plusEquals( const unsigned long aa, const unsigned long bb, const unsigned long cc, const unsigned long dd, const T value ) {
            const unsigned long index = getIndex( aa, bb, cc, dd );
            return m_data[index] += value;
        }

        inline T plusEquals( const unsigned long x, const unsigned long y, const unsigned long z, const T value ) {
            const unsigned long index = getIndex( x, y, z );
            return m_data[index] += value;
        }

        inline T plusEquals( const unsigned long x, const unsigned long y, const T value ) {
            const unsigned long index = getIndex( x, y );
            return m_data[index] += value;
        }

        inline T plusEquals( const unsigned long x, const T value ) {
            const unsigned long index = getIndex( x );
            return m_data[index] += value;
        }

        inline T dot( const TMatrix &matrix ) const { // Calculate the dot product: scalar = lhs (dot) rhs;
            if( isEmpty()) {
                log_error( "empty matrix" );
                return 0;
            }
            if( matrix.m_count != m_count ) {
                log_error( "Matricies not the same size" );
                return 0;
            }
            const T *       lhs = m_data;
            const T * const end = m_end;
            const T *       rhs = matrix.m_data;
            T result = 0;
            while( lhs < end ) {
                result += *lhs++ * *rhs++;
            }
            return result;
        }
        
        inline T min( void ) const {                // Return minimum value.
            if( isEmpty()) {
                log_error( "empty matrix" );
                return 0;
            }
            const T *      data = m_data;
            const T * const end = m_end;
            
            T result = *data++;
            while( data < end ) {
                if( *data < result ) {
                    result = *data;
                }
                data++;
            }
            return result;
        }
        
        inline T max( void ) const {                // Return maximum value.
            if( isEmpty()) {
                log_error( "empty matrix" );
                return 0;
            }
            const T *      data = m_data;
            const T * const end = m_end;
            
            T result = *data++;
            while( data < end ) {
                if( *data > result ) {
                    result = *data;
                }
                data++;
            }
            return result;
        }
 
        inline T sum( void ) const {                    // Return sum of matrix elements.
            if( isEmpty()) {
                log_error( "empty matrix" );
                return 0;
            }
            const T *      data = m_data;
            const T * const end = m_end;
            
            T result = *data++;
            while( data < end ) {
                result += *data++;
            }
            return result;
        }
        
        inline void add( const T value ) const {          // Add a scalar to each element.
            if( isEmpty()) {
                log_error( "empty matrix" );
                return ;
            }
            const T *      data = m_data;
            const T * const end = m_end;
            
            while( data < end ) {
                *data++ += value;
            }
            return;
        }


        inline void multiply( const T value ) const {     // Multiple each element by a scalar.
            if( isEmpty()) {
                log_error( "empty matrix" );
                return ;
            }
            const T *      data = m_data;
            const T * const end = m_end;
            
            while( data < end ) {
                *data++ *= value;
            }
            return;
        }
        
    };  // class TMatrix
    
    template <typename T> struct TTrainable {       // Tuple of weight Matrix and gradiant Matrix.
              T *weightStart;                       // Convienience pointers to the start and end of data arrays
        const T *weightEnd;
              T *gradiantStart;
        const T *gradiantEnd;
              T  l1_decay_mul;
              T  l2_decay_mul;

        TTrainable( void ):
        weightStart( 0 ), weightEnd( 0 ), gradiantStart( 0 ), gradiantEnd( 0 ),
        l1_decay_mul( 1.0 ), l2_decay_mul( 1.0 ) {
        }
        
        TTrainable( TMatrix< T > *weights, TMatrix< T > *gradiants ) :
        weightStart( 0 ), weightEnd( 0 ), gradiantStart( 0 ), gradiantEnd( 0 ),
        l1_decay_mul( 1.0 ), l2_decay_mul( 1.0 ) {
            setWeight(   weights   );
            setGradiant( gradiants );
        }
        
        ~TTrainable( void ) {
            weightStart   = 0;
            weightEnd     = 0;
            gradiantStart = 0;
            gradiantEnd   = 0;
        }
        
        inline bool ok( void ) const {
            return weightStart != 0 && weightStart < weightEnd && gradiantStart != 0 && gradiantStart < gradiantEnd;
        }
        
        inline unsigned long weightCount( void ) const {
            if( weightEnd <= weightStart ) {
                return 0;
            }
            return (unsigned long) ( weightEnd - weightStart );             // Count of Ts
        }

        inline unsigned long gradiantCount( void ) const {
            if( gradiantEnd <= gradiantStart ) {
                return 0;
            }
            return (unsigned long) ( gradiantEnd - gradiantStart );         // Count of Ts
        }

        void setWeight( TMatrix< T > *matrix ) {
            if( matrix != 0 ) {
                weightStart = matrix->data();
                weightEnd   = matrix->end();
            } else {
                weightStart = 0;
                weightEnd   = 0;
            }
        }

        void setGradiant( TMatrix< T > *matrix ) {
            if( matrix != 0 ) {
                gradiantStart = matrix->data();
                gradiantEnd   = matrix->end();
            } else {
                gradiantStart = 0;
                gradiantEnd   = 0;
            }
        }

    };  // struct TTrainable
    
    typedef TMatrix< DNN_NUMERIC >  Matrix;     // Typically double
    typedef TMatrix< DNN_INTEGER > DMatrix;     // Typically long
    
    typedef TTrainable< DNN_NUMERIC >  Trainable;
    typedef TTrainable< DNN_INTEGER > DTrainable;
    
    inline bool matrixBad( const  Matrix * const matrix ) { return matrix == 0 || !matrix->ok(); }
    inline bool matrixBad( const DMatrix * const matrix ) { return matrix == 0 || !matrix->ok(); }
    
    bool softmax( Matrix &dst, const Matrix &src );
    
    Matrix *gaussianKernel( const unsigned long side, const DNN_NUMERIC sigma = 1.0 );   // Typically 5 or 7
    
    Matrix *kernelOperation( const Matrix &src, const Matrix &kernel, const unsigned long stride );
        
    bool subsample( Matrix &dst, const Matrix &src, const unsigned long dx, const unsigned long dy );
    
}   // namespace tfs

#endif /* Matrix_h */
